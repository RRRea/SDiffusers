from fastapi import FastAPI, Query
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from DeepCache import DeepCacheSDHelper
import torch
import random
import numpy as np
from PIL import Image
from starlette.responses import StreamingResponse
import io
import os
from RealESRGAN import RealESRGAN
import cv2
import gc

app = FastAPI()
model_name = "cagliostrolab/animagine-xl-3.1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["USE_TORCH_COMPILE"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_SEED = np.iinfo(np.int32).max
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

pipe = DiffusionPipeline.from_pretrained(
    model_name,
    custom_pipeline="lpw_stable_diffusion_xl",
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False,
)
#Lower cache_branch_id: This speeds up processing but might make the results less accurate.
#Larger cache_interval: Faster, but the image quality might suffer.
helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=2,
    cache_branch_id=4,
)
helper.enable()
pipe.to(device)

DEFAULT_CLIP = 2
DEFAULT_RANDOMIZE_SEED = True
DEFAULT_SEED = 0
UPSCALER = True
DEFAULT_PROMPT = ", masterpiece, best quality, very aesthetic, absurdres"
DEFAULT_NEGATIVE_PROMPT = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 1216
DEFAULT_GUIDANCE_SCALE = 7
DEFAULT_NUM_INFERENCE_STEPS = 20

@app.post("/generate_image/")
async def generate_image(
    clip_skip: int = DEFAULT_CLIP,
    randomize_seed: bool = DEFAULT_RANDOMIZE_SEED,
    seed: int = DEFAULT_SEED,
    use_upscaler: bool = UPSCALER,
    upscaler_types: str = Query("RealESRGAN_x4", enum=["RealESRGAN_x2","RealESRGAN_x4","RealESRGAN_x8"]),
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
):
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    image = pipe(
        clip_skip=clip_skip,
        randomize_seed=randomize_seed,
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]

    image_path = "./image.png"
    image.save(image_path, format="PNG")

    if use_upscaler:
      if upscaler_types == "RealESRGAN_x2":
          model = RealESRGAN(device, scale=2)
          model.load_weights("weights/RealESRGAN_x2.pth", download=True)
      elif upscaler_types == "RealESRGAN_x4":
          model = RealESRGAN(device, scale=4)
          model.load_weights("weights/RealESRGAN_x4.pth", download=True)
      elif upscaler_types == "RealESRGAN_x8":
          model = RealESRGAN(device, scale=8)
          model.load_weights("weights/RealESRGAN_x8.pth", download=True)
      else:
          return {"error": "Invalid upscaler type specified"}
      path_to_image = './image.png'
      image = Image.open(path_to_image).convert('RGB')
      sr_image = model.predict(image)
      sr_image.save('./upscaled_image.png')
      image_data = io.BytesIO()
      sr_image.save(image_data, format="PNG")
      image_data.seek(0)
      return StreamingResponse(image_data, media_type="image/png")

    image_data = io.BytesIO()
    image.save(image_data, format="PNG")
    image_data.seek(0)
    return StreamingResponse(image_data, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
