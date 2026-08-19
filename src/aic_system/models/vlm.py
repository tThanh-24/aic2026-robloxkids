"""Q&A answering and candidate verification with Qwen2-VL (HF transformers, fp16).

Memory budget on a 24 GB RTX 3090: Qwen2-VL-7B fp16 ≈ 16 GB. CLIP text
encoder (~350 MB) coexists fine; FAISS stays on CPU by design. If loading
still OOMs, the constructor automatically retries with the configured
fallback model (Qwen2-VL-2B ≈ 5 GB).

Two entry points:
  - answer_question(image | [images], question): free-text answer, used by
    the Q&A runner. Accepts multiple keyframes so temporal/counting
    questions see more context than a single frame.
  - verify(image | [images], statement): yes/no judgement with confidence,
    used to re-rank KIS candidates and validate TRAKE event frames.

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

_VERIFY_INSTRUCTIONS = {
    "vi": "Hình ảnh có khớp với mô tả trên không? Chỉ trả lời một từ: Có hoặc Không.",
    "en": "Does the image match the description above? Answer one word only: Yes or No.",
}

_YES_WORDS = {"en": {"yes", "correct", "true"}, "vi": {"có", "đúng", "vàng", "vâng"}}
_NO_WORDS = {"en": {"no", "not", "incorrect", "false"}, "vi": {"không", "sai"}}


def parse_yes_no(text: str, lang: str = "en") -> bool | None:
    """First-word yes/no parse of a VLM reply. None when undecidable."""
    first = (text or "").strip().lower().split(" ")[:1]
    first = first[0].strip(".,!?;:") if first else ""
    if first in _YES_WORDS.get(lang, ()):
        return True
    if first in _NO_WORDS.get(lang, ()):
        return False
    # Cross-language fallback: a Vietnamese "Có"/"Không" under lang="en" etc.
    for words in _YES_WORDS.values():
        if first in words:
            return True
    for words in _NO_WORDS.values():
        if first in words:
            return False
    return None


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

    # ------------------------------------------------------------------
    # Core generation (shared by answering and verification)
    # ------------------------------------------------------------------
    def _generate(self, pil_images: list, prompt_text: str) -> tuple[str, float]:
        """(images, prompt) -> (generated_text, mean token probability)."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * len(pil_images)
                + [{"type": "text", "text": prompt_text}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=pil_images, return_tensors="pt", padding=True
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
        return answer, confidence

    @staticmethod
    def _open_images(image_paths: Path | str | list) -> list:
        from PIL import Image

        paths = [image_paths] if isinstance(image_paths, (str, Path)) else list(image_paths)
        images = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except (FileNotFoundError, OSError) as e:
                print(f"  [vlm] unreadable image {p}: {e}")
        return images

    # ------------------------------------------------------------------
    # Q&A answering
    # ------------------------------------------------------------------
    def answer_question(
        self, image_paths: Path | str | list, question: str, lang: str = "vi"
    ) -> tuple[str, float]:
        """(image(s), question) -> (answer_text, confidence).

        Passing several keyframes of one video gives the model temporal
        context (order/count questions). Confidence is the geometric mean
        token probability of the generated answer -- a proxy for how sure
        the model is, used only to rank candidate frames relative to each
        other.
        """
        images = self._open_images(image_paths)
        if not images:
            return "", 0.0
        instruction = _ANSWER_INSTRUCTIONS.get(lang, _ANSWER_INSTRUCTIONS["en"])
        return self._generate(images, f"{question}\n{instruction}")

    # ------------------------------------------------------------------
    # Candidate verification (KIS re-rank, TRAKE frame validation)
    # ------------------------------------------------------------------
    def verify(
        self, image_paths: Path | str | list, statement: str, lang: str = "vi"
    ) -> tuple[bool, float]:
        """'Is this image described by {statement}?' -> (matches, confidence).

        Undecidable replies parse as (False, confidence): an unverifiable
        candidate must never outrank an explicitly verified one.
        """
        images = self._open_images(image_paths)
        if not images:
            return False, 0.0
        instruction = _VERIFY_INSTRUCTIONS.get(lang, _VERIFY_INSTRUCTIONS["en"])
        answer, conf = self._generate(images, f"{statement}\n{instruction}")
        return (parse_yes_no(answer, lang=lang) is True), conf

    def answer_batch(
        self, items: list[tuple[Path, str]], lang: str = "vi"
    ) -> list[tuple[str, float]]:
        """Sequential inference (batching images with Qwen2-VL's variable
        resolution buys little at top_k_for_vqa ~ 20 candidates/query)."""
        return [self.answer_question(img, q, lang) for img, q in items]
