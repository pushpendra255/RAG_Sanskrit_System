"""Grounded answer generation for the Sanskrit RAG pipeline."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import List, Sequence, Set, Tuple

try:
    from utils import (
        FALLBACK_ANSWER,
        contains_devanagari,
        get_focus_tokens,
        get_query_hint_tokens,
        prepare_query_variants,
        romanized_anchor_score,
        romanized_overlap_score,
        token_anchor_score,
    )
except ImportError:
    from .utils import (
        FALLBACK_ANSWER,
        contains_devanagari,
        get_focus_tokens,
        get_query_hint_tokens,
        prepare_query_variants,
        romanized_anchor_score,
        romanized_overlap_score,
        token_anchor_score,
    )


class LLMGenerator:
    """Hybrid generator with a CPU LLM path and a grounded extractive fallback."""

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "google/flan-t5-small",
        cache_dir: str | None = None,
        max_new_tokens: int = 96,
    ):
        self.backend = backend
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.runtime_backend = "extractive"
        self.model = None
        self.tokenizer = None
        self._load_backend()

    def _load_backend(self) -> None:
        """Load the preferred generator backend, falling back safely when needed."""
        if self.backend in {"auto", "transformers"}:
            try:
                self._load_transformers_model()
                self.runtime_backend = "transformers"
                return
            except Exception as exc:
                print(f"Warning: could not load transformers generator ({exc}).")
                if self.backend == "transformers":
                    raise

        self.runtime_backend = "extractive"

    def _load_transformers_model(self) -> None:
        """Load a lightweight seq2seq model for CPU-only inference."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.cache_dir:
            os.environ.setdefault("TRANSFORMERS_CACHE", self.cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def normalize_token(self, token: str) -> str:
        """Normalize a Sanskrit token for simple lexical matching."""
        token = token.strip().lower()
        token = re.sub(r"[^\u0900-\u097Fa-z0-9]", "", token)

        suffixes = ["ः", "म्", "ं", "ा", "े", "ि", "ु"]
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix):
                token = token[: -len(suffix)]
                break

        return token

    def _normalize_for_match(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text or "").lower()
        text = re.sub(r"[^\u0900-\u097Fa-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _split_context_sentences(self, context: str) -> List[str]:
        if not context:
            return []

        stripped_context = re.sub(r"\[[^\]]+\]\s*", "", context)
        stripped_context = stripped_context.replace("\n", " ")
        stripped_context = re.sub(r"\s+", " ", stripped_context).strip()

        raw_sentences = re.split(r"[।॥.?]+", stripped_context)
        sentences = []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if 18 <= len(sentence) <= 320:
                sentences.append(sentence)
        return sentences

    def _detect_query_type(self, query: str) -> str:
        normalized_query = self._normalize_for_match(query)

        if any(token in normalized_query for token in ["किम्", "अकरोत्", "क्या", "कथम्"]):
            return "action"

        if any(token in normalized_query for token in ["सन्देश", "शिक्षा", "तात्पर्य"]):
            return "moral"

        return "fact"

    def _score_sentence(self, sentence: str, query_tokens: Set[str], query_type: str, focus_tokens: List[str], query: str) -> int:
        sentence_normalized = self._normalize_for_match(sentence)
        if "इति" in sentence_normalized and ("आदिशति" in sentence_normalized or "वदति" in sentence_normalized):
            return -999

        sentence_tokens = {
            self.normalize_token(token)
            for token in sentence_normalized.split()
            if len(token.strip()) > 1
        }
        score = len(query_tokens.intersection(sentence_tokens))

        for query_token in query_tokens:
            if query_token and query_token in sentence_normalized:
                score += 1

        anchor_score = token_anchor_score(focus_tokens, sentence)
        score += int(anchor_score * 6)
        primary_anchor_score = token_anchor_score(focus_tokens[:1], sentence)
        score += int(primary_anchor_score * 5)

        focus_position_boost = self._focus_position_boost(sentence_normalized, focus_tokens)
        score += focus_position_boost

        if query_type == "action" and any(
            marker in sentence_normalized
            for marker in ["गच्छ", "न्यस्य", "कर्ष", "प्रवह", "वद", "उक्त"]
        ):
            score += 2

        if query_type == "action" and any(marker in sentence_normalized for marker in ["अभिधाव", "आक्रोश", "कोपेन"]):
            score -= 2
        if query_type == "action" and sentence.strip().startswith('"'):
            score -= 4
        if query_type == "action" and "आनय" in sentence_normalized and '"' in sentence:
            score -= 3
        if any(marker in self._normalize_for_match(query) for marker in ["नष्ट", "हत", "नाश"]):
            if any(marker in sentence_normalized for marker in ["हत", "नाश", "मया", "आदाय"]):
                score += 4
        if any(marker in self._normalize_for_match(query) for marker in ["सहाय्य", "help"]):
            if any(marker in sentence_normalized for marker in ["दत्त", "आनीत", "सुभाषित", "उपाय"]):
                score += 4

        if query_type == "moral" and any(
            marker in sentence_normalized
            for marker in ["विनश्यति", "संसर्गात्", "उद्यम", "साहाय्य", "वरम्"]
        ):
            score += 2

        if sentence.count('"') >= 2:
            score -= 1

        return score

    def _select_best_sentence(self, context: str, query: str) -> Tuple[str, int]:
        if not context or not query:
            return "", 0

        if not contains_devanagari(query):
            return self._select_best_sentence_for_latin_query(context, query)

        query_tokens = {
            self.normalize_token(token)
            for token in self._normalize_for_match(query).split()
            if len(token.strip()) > 1
        }
        if not query_tokens:
            return "", 0

        query_type = self._detect_query_type(query)
        focus_tokens = list(dict.fromkeys(get_focus_tokens(query) + get_query_hint_tokens(query)))
        best_sentence = ""
        best_score = 0

        for sentence in self._split_context_sentences(context):
            score = self._score_sentence(sentence, query_tokens, query_type, focus_tokens, query)
            if score == -999:
                continue
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence, best_score

    def _select_best_sentence_for_latin_query(self, context: str, query: str) -> Tuple[str, int]:
        """Score context sentences against transliterated queries."""
        best_sentence = ""
        best_score = 0.0
        focus_tokens = list(dict.fromkeys(get_focus_tokens(query) + get_query_hint_tokens(query)))

        for sentence in self._split_context_sentences(context):
            score = (
                0.55 * romanized_overlap_score(query, self._romanize(sentence))
                + 0.45 * romanized_anchor_score(focus_tokens, self._romanize(sentence))
            )
            score += 0.30 * romanized_anchor_score(focus_tokens[:1], self._romanize(sentence))
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence, int(best_score * 10)

    def _build_prompt(self, context: str, query: str, extractive_hint: str) -> str:
        """Construct a short grounded prompt for the CPU seq2seq model."""
        prompt = [
            "Answer the Sanskrit question using only the provided context.",
            f"If the answer is missing, reply exactly: {FALLBACK_ANSWER}",
            "Prefer a short Sanskrit answer copied or paraphrased from the context.",
            f"Question: {query}",
            f"Context: {context}",
        ]
        if extractive_hint:
            prompt.append(f"Relevant sentence: {extractive_hint}")
        prompt.append("Answer:")
        return "\n".join(prompt)

    def _generate_with_transformers(self, context: str, query: str, extractive_hint: str) -> str:
        """Run generation through a lightweight seq2seq model on CPU."""
        prompt = self._build_prompt(context, query, extractive_hint)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=2,
            early_stopping=True,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return re.sub(r"\s+", " ", text)

    def generate(self, context: str, query: str, query_variants: Sequence[str] | None = None) -> str:
        """Produce a grounded answer from retrieved context."""
        if not context or not query:
            return FALLBACK_ANSWER

        variants = list(query_variants or prepare_query_variants(query))
        extractive_candidates = []
        for variant in variants or [query]:
            sentence, score = self._select_best_sentence(context, variant)
            if sentence:
                extractive_candidates.append((sentence, score))

        extractive_candidates.sort(key=lambda item: item[1], reverse=True)
        extractive_hint = extractive_candidates[0][0] if extractive_candidates else ""

        if self.runtime_backend == "transformers" and self.model is not None and self.tokenizer is not None:
            try:
                answer = self._generate_with_transformers(context, query, extractive_hint)
                if self._is_acceptable_answer(answer):
                    return self._postprocess_answer(answer)
            except Exception as exc:
                print(f"Warning: transformers generation failed, using extractive fallback ({exc}).")

        if extractive_hint:
            return self._postprocess_answer(extractive_hint.strip())

        return FALLBACK_ANSWER

    @staticmethod
    def _romanize(text: str) -> str:
        """Romanize a Sanskrit sentence for transliterated query scoring."""
        try:
            from indic_transliteration import sanscript
        except ImportError:
            return text

        try:
            return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
        except Exception:
            return text

    @staticmethod
    def _is_acceptable_answer(answer: str) -> bool:
        """Reject obviously broken seq2seq outputs before returning them."""
        if not answer:
            return False

        cleaned = re.sub(r"\s+", " ", answer).strip()
        if cleaned == FALLBACK_ANSWER:
            return True

        if len(cleaned) < 10:
            return False

        if cleaned.startswith("[") or cleaned.startswith("Context:"):
            return False

        return True

    @staticmethod
    def _postprocess_answer(answer: str) -> str:
        """Trim common retrieval artifacts from otherwise correct answers."""
        cleaned = re.sub(r"\s+", " ", answer or "").strip()

        if "स्वस्ति श्री" in cleaned:
            cleaned = cleaned.split("स्वस्ति श्री", 1)[0].strip(" ./|")

        cleaned = re.sub(r"\s+स्$", "", cleaned).strip()
        cleaned = cleaned.rstrip("/| ")
        return cleaned or FALLBACK_ANSWER

    @staticmethod
    def _focus_position_boost(sentence_normalized: str, focus_tokens: List[str]) -> int:
        """Prefer sentences that mention the main query entity early."""
        if not focus_tokens:
            return 0

        first_words = sentence_normalized.split()[:6]
        if not first_words:
            return 0

        window_text = " ".join(first_words)
        for focus_token in focus_tokens:
            if focus_token and focus_token in window_text:
                return 3

        return 0


def format_context(results: Sequence[Tuple[dict, float]], max_context_length: int = 1600, max_chunks: int = 4) -> str:
    """Format retrieved chunks into a compact grounded context block."""
    if not results:
        return ""

    context_parts: List[str] = []
    total_length = 0

    for chunk, _score in results[:max_chunks]:
        content = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
        if not content:
            continue

        label = chunk.get("section_title") or chunk.get("source", "document")
        formatted = f"[{chunk.get('source', 'document')} | {label}] {content}"
        if total_length + len(formatted) > max_context_length:
            break

        context_parts.append(formatted)
        total_length += len(formatted)

    return "\n".join(context_parts)
