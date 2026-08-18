"""CLIP text encoder matching the organizer-provided image features.

The dataset ships CLIP ViT-B/32 image embeddings (clip-features-32-aic25-b1,
dim 512). Text queries must be encoded with the TEXT tower of the *same*
model or the dot products are meaningless. Uses the HuggingFace `CLIPModel`
(no extra dependency -- transformers is already required for the VLM).

If retrieval results look like noise on the real data, the likely cause is
a features/text mismatch; try `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
(configs: models.clip_text.name) before debugging anything else.
"""
from __future__ import annotations

import numpy as np

_DEVICE_FALLBACK_NOTE = "cuda unavailable, running CLIP text encoder on cpu"


class CLIPTextEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        import torch
        from transformers import AutoTokenizer, CLIPModel

        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"  [clip] {_DEVICE_FALLBACK_NOTE}")
            device = "cpu"
        self.device = device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        if device.startswith("cuda"):
            self.model = self.model.half()

    def encode(self, text: str) -> np.ndarray:
        """Single text -> (512,) float32 L2-normalized vector."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """N texts -> (N, 512) float32 L2-normalized vectors (cpu numpy)."""
        torch = self.torch
        with torch.no_grad():
            tokens = self.tokenizer(
                texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)
            feats = self.model.get_text_features(**tokens)  # (N, 512)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats.float().cpu().numpy().astype(np.float32, copy=False)
