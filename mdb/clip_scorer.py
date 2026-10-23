"""CLIPScore = 100 * max(cos(image, text), 0) using OpenCLIP."""
import os

import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_SCORER = os.environ.get("MDB_CLIP", "ViT-H-14:laion2b_s32b_b79k")


class ClipScorer:
    def __init__(self, spec: str = DEFAULT_SCORER, device: str | None = None):
        import open_clip
        arch, pretrained = spec.split(":")
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.tokenizer = open_clip.get_tokenizer(arch)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else torch.autocast("cpu", enabled=False)

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        if self.device == "cuda":
            x = x.half()
        with self._autocast():
            e = self.model.encode_image(x)
        return F.normalize(e.float(), dim=-1)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        t = self.tokenizer(texts).to(self.device)
        with self._autocast():
            e = self.model.encode_text(t)
        return F.normalize(e.float(), dim=-1)

    def score(self, images: list[Image.Image], texts: list[str]) -> list[float]:
        cos = (self.embed_images(images) * self.embed_texts(texts)).sum(-1)
        return cos.cpu().tolist()


def clip_score(cos: float) -> float:
    return 100.0 * max(cos, 0.0)
