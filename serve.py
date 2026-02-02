import gc
import argparse
import asyncio
from io import BytesIO
from pathlib import Path
from time import time

import yaml
import torch
import uvicorn
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI,  UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

import random
import numpy as np
from image_edit.qwen_edit_module import QwenEditModule
from config import settings

REQUIRED_MODELS = {
    "microsoft/TRELLIS.2-4B",
    "microsoft/TRELLIS-image-large",
    "ZhengPeng7/BiRefNet",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def load_model_versions() -> dict[str, str]:
    """Load pinned model versions from model_versions.yml."""
    versions_file = Path(__file__).parent / "model_versions.yml"
    with open(versions_file) as f:
        data = yaml.safe_load(f)["huggingface"]
    
    # Extract revisions and validate
    model_versions = {k: v["revision"] for k, v in data.items()}
    
    if missing := REQUIRED_MODELS - model_versions.keys():
        raise ValueError(f"Missing required models in model_versions.yml: {missing}")
    
    return model_versions


def get_args() -> argparse.Namespace:
    """ Function for getting arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10006)
    return parser.parse_args()


def clean_vram() -> None:
    """ Function for cleaning VRAM. """
    gc.collect()
    torch.cuda.empty_cache()


executor = ThreadPoolExecutor(max_workers=1)

class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    logger.info("Loading Trellis 2 generator models ...")
    try:
        model_versions = load_model_versions()
        logger.info(f"Loaded pinned revisions for {len(model_versions)} models")
        
        app.state.trellis_generator = Trellis2ImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B",
            model_versions,
        )
        app.state.trellis_generator.to("cuda")
        
                # Load Qwen Edit Module if enabled
        if settings.enable_image_edit:
            logger.info("Loading Qwen Edit Module ...")
            app.state.qwen_edit = QwenEditModule(settings)
            app.state.qwen_edit.startup()
            logger.success("Qwen Edit Module loaded successfully")
        else:
            app.state.qwen_edit = None
            logger.info("Image editing is disabled")

    except Exception as e:
        logger.exception(f"Exception during model loading: {e}")
        raise SystemExit("Model failed to load → exiting server")

    yield
    
    # Cleanup
    if hasattr(app.state, 'qwen_edit') and app.state.qwen_edit is not None:
        app.state.qwen_edit.shutdown()


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan

def set_seed(seed: int) -> None:
    """ Function for setting global seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def generation_block(prompt_image: Image.Image, seed: int = -1):
    """ Function for 3D data generation using provided image"""

    t_start = time()
       
    # Set seed if provided
    if seed >= 0:
        set_seed(seed)
    
    # Apply image editing if enabled
    if settings.enable_image_edit and app.state.qwen_edit is not None:
        # logger.info("Applying image editing before 3D generation...")
        t_edit_start = time()
        try:
            prompt_image = app.state.qwen_edit.edit_image(
                prompt_image=prompt_image,
                seed=seed if seed >= 0 else None
            )
            t_edit_end = time()
            logger.info(f"Image editing took: {(t_edit_end - t_edit_start):.2f} secs.")
        except Exception as e:
            logger.error(f"Image editing failed: {e}. Proceeding with original image.")
    # rad_num = random.randint(0, 10000)
    # prompt_image.save(f"prompt_image_{rad_num}.png")
    mesh = app.state.trellis_generator.run(image=prompt_image, seed=seed, pipeline_type="512")[0]
    mesh.simplify()

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=1024,
        remesh=True,
        remesh_band=1,
        remesh_project=0.9,
        verbose=False
    )

    buffer = BytesIO()
    glb.export(buffer, extension_webp=False, file_type="glb")
    buffer.seek(0)

    t_get_model = time()
    logger.debug(f"Model Generation took: {(t_get_model - t_start)} secs.")

    clean_vram()

    t_gc = time()
    logger.debug(f"Garbage Collection took: {(t_gc - t_get_model)} secs")

    return buffer


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> Response:
    """ Generates a 3D model as GLB file """

    logger.info("Task received. Prompt-Image")

    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(executor, generation_block, prompt_image, seed)
    buffer_size = len(buffer.getvalue())
    buffer.seek(0)
    logger.info(f"Task completed.")

    async def generate_chunks():
        chunk_size = 1024 * 1024  # 1 MB
        while chunk := buffer.read(chunk_size):
            yield chunk

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={"Content-Length": str(buffer_size)}
    )


@app.get("/version", response_model=str)
async def version() -> str:
    """ Returns current endpoint version."""
    return app.version


@app.get("/health")
def health_check() -> dict[str, str]:
    """ Return if the server is alive """
    return {"status": "healthy"}


if __name__ == "__main__":
    args: argparse.Namespace  = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
