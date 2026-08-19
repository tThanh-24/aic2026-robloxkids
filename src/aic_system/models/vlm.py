"""Q&A answering with Qwen2-VL (HuggingFace transformers, fp16).

Memory budget on a 24 GB RTX 3090: Qwen2-VL-7B fp16 ≈ 16 GB. CLIP text
encoder (~350 MB) coexists fine; FAISS stays on CPU by design. If loading
still OOMs, the constructor automatically retries with the configured
fallback model (Qwen2-VL-2B ≈ 5 GB).

Answers are judged SEMANTICALLY (per the competition evaluator), so the
prompt asks for a short natural answer in the query's own language --
no need to force a canonical single-word format. We still keep answers
terse because the CSV caps them at 100 characters.
"""
from __future__ import annotations

from pathlib import Path

_ANSWER_INSTRUCTIONS = {
    "vi": "Trả lời ngắn gọn bằng tiếng Việt, không giải thích.",
    "en": "Answer briefly in English, no explanation.",
}


class VLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        fallback_name: str | None = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 32,
    ):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

        def _load(name: str):
            model_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                name, torch_dtype=model_dtype, device_map="auto"
            )
            model.eval()
            return model, AutoProcessor.from_pretrained(name)

        try:
            self.model, self.processor = _load(model_name)
        except torch.cuda.OutOfMemoryError:
            if not fallback_name:
                raise
            print(f"  [vlm] OOM loading {model_name}; falling back to {fallback_name}")
            self.torch.cuda.empty_cache()
            self.model_name = fallback_name
            self.model, self.processor = _load(fallback_name)

    def answer_question(self, image_path: Path, question: str, lang: str = "vi") -> tuple[str, float]:
        """(image, question) -> (answer_text, confidence).

        Confidence is the geometric mean token probability of the generated
        answer -- a proxy for how sure the model is, used only to rank
        candidate frames relative to each other.
        """
        import logging

        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"  [vlm] unreadable image {image_path}: {e}")
            return "", 0.0

        instruction = _ANSWER_INSTRUCTIONS.get(lang, _ANSWER_INSTRUCTIONS["en"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}\n{instruction}"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        ).to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        token_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(token_ids, skip_special_tokens=True).strip()
        # Mean token probability over actually-generated tokens (len >= 1).
        probs = self.torch.softmax(
            self.torch.stack(out.scores, dim=1).float(), dim=-1
        )  # (1, T, V)
        gen_probs = probs[0, self.torch.arange(len(token_ids), device=probs.device), token_ids.to(probs.device)]
        confidence = float(self.torch.exp(self.torch.log(gen_probs.clamp_min(1e-9)).mean()))

        logging.getLogger(__name__).debug("vlm answer=%r conf=%.3f", answer, confidence)
        return answer, confidence

    def answer_batch(
        self, items: list[tuple[Path, str]], lang: str = "vi"
    ) -> list[tuple[str, float]]:
        """Sequential inference (batching images with Qwen2-VL's variable
        resolution buys little at top_k_for_vqa ~ 20 candidates/query)."""
        return [self.answer_question(img, q, lang) for img, q in items]
