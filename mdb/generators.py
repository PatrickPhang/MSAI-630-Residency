"""Local text-to-image backends: SDXL-Turbo and SD 1.5."""
import gc

import torch


class SDXLTurbo:
    name = "sdxl_turbo"

    def __init__(self, steps: int = 4, size: int = 512):
        from diffusers import AutoPipelineForText2Image
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.size = steps, size

    @torch.no_grad()
    def __call__(self, prompt: str, seed: int, guidance: float = 0.0):
        g = torch.Generator("cuda").manual_seed(int(seed))
        out = self.pipe(prompt=prompt, num_inference_steps=self.steps, guidance_scale=guidance,
                        width=self.size, height=self.size, generator=g)
        return out.images[0], "ok", f"steps={self.steps};cfg={guidance}"


class SD15:
    name = "sd15"

    def __init__(self, steps: int = 25):
        from diffusers import StableDiffusionPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.steps = steps

    @torch.no_grad()
    def __call__(self, prompt: str, seed: int, guidance: float = 7.5):
        g = torch.Generator("cuda").manual_seed(int(seed))
        out = self.pipe(prompt=prompt, num_inference_steps=self.steps, guidance_scale=guidance, generator=g)
        flagged = bool(out.nsfw_content_detected and out.nsfw_content_detected[0])
        if flagged:
            return None, "refused", "sd15 safety checker flagged the image"
        return out.images[0], "ok", f"steps={self.steps};cfg={guidance}"


BACKENDS = {"sdxl_turbo": SDXLTurbo, "sd15": SD15}


def load_backend(name: str, **kw):
    if name not in BACKENDS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(BACKENDS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. In Colab: Runtime > Change runtime type > T4 GPU.")
    return BACKENDS[name](**kw)


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
