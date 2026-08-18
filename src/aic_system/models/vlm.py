"""Q&A answering with Qwen2-VL (HuggingFace transformers, fp16).

Memory budget on a 24 GB RTX 3090: Qwen2-VL-7B fp16 ≈ 16 GB. CLIP text
encoder (~350 MB) coexists fine; FAISS stays on CPU by design. If loading
still OOMs, the constructor automatically retries with the configured
fallback model (Qwen2-VL-2B ≈ 5 GB).

Answers are judged SEMANTICALLY (per the competition evaluator), so the
prompt asks for a short natural answer in the query's own language --
no need to force a canonical single-word format. We still keep answers
terse because the CSV caps them at 100 characters.
Besides answering, the same model powers the rerank stages of KIS/TRAKE:
  - verify(image, statement): yes/no frame relevance as a [0, 1] score
  - split_events(text): text-only TRAKE event splitting (the regex fallback
    stays in retrieval.search for when the VLM is unavailable)
"""
from __future__ import annotations

import re
from pathlib import Path

_ANSWER_INSTRUCTIONS = {
    "vi": "Trả lời ngắn gọn bằng tiếng Việt, không giải thích.",
    "en": "Answer briefly in English, no explanation.",
}

_VERIFY_INSTRUCTION = (
    "Does this image match the description above? Answer YES or NO only."
)

_EVENT_SPLIT_PROMPT = (
    "A query describes events that happen in a video, in temporal order. "
    "Split it into the distinct events, one short line each, in order, "
    "without numbering. If it describes a single moment or scene, output "
    "exactly one line.\n\nQuery: {query}"
)


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
        gen_probs = probs[0, torch.arange(len(token_ids), device=probs.device), token_ids.to(probs.device)]
        confidence = float(self.torch.exp(self.torch.log(gen_probs.clamp_min(1e-9)).mean()))

        logging.getLogger(__name__).debug("vlm answer=%r conf=%.3f", answer, confidence)
        return answer, confidence

    def verify(self, image_path: Path, statement: str) -> float:
        """(image, statement) -> frame relevance in [0, 1].

        Asks for a YES/NO verdict and reads the FIRST generated token's
        probability mass on yes-starting vs no-starting vocabulary:
        p_yes / (p_yes + p_no). Normalizing by the yes+no mass makes the
        score robust to the model ignoring the format; when it answers
        neither (or the image is unreadable) the score is a neutral 0.5,
        so an unverifiable frame keeps its retrieval position.
        """
        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"  [vlm] unreadable image {image_path}: {e}")
            return 0.5

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{statement}\n{_VERIFY_INSTRUCTION}"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        ).to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        probs = self.torch.softmax(out.scores[0][0].float(), dim=-1)
        top = self.torch.topk(probs, k=20)
        p_yes = p_no = 0.0
        for prob, tid in zip(top.values.tolist(), top.indices.tolist()):
            token = self.processor.tokenizer.decode([tid]).strip(" .,:;!").lower()
            if token.startswith("yes"):
                p_yes += prob
            elif token.startswith("no"):
                p_no += prob
        if p_yes + p_no < 1e-6:
            return 0.5
        return p_yes / (p_yes + p_no)

    def split_events(self, text: str) -> list[str] | None:
        """Text-only: split a TRAKE query into its events, one per line.

        Returns None on generation/parse failure so the caller keeps its
        regex fallback; callers should still sanity-bound the event count
        (a wrong N invalidates every emitted row).
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": _EVENT_SPLIT_PROMPT.format(query=text)}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        answer = self.processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        events = []
        for line in answer.splitlines():
            line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip(" \t")
            if len(line) >= 3:
                events.append(line)
        return events or None

    def answer_batch(
        self, items: list[tuple[Path, str]], lang: str = "vi"
    ) -> list[tuple[str, float]]:
        """Sequential inference (batching images with Qwen2-VL's variable
        resolution buys little at top_k_for_vqa ~ 20 candidates/query)."""
        return [self.answer_question(img, q, lang) for img, q in items]
