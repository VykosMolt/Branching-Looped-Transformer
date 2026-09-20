"""Experimental MMLU science MCQ parser candidates for v3 repair.

This module is local to manual probes. It does not replace historical or
production parser logic. Callers must explicitly choose one of these parsers.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


OPTION_LETTERS = ("A", "B", "C", "D", "E")
FINAL_MARKERS = (
    "final answer",
    "answer:",
    "the answer is",
    "answer is",
    "therefore",
    "thus",
    "so the answer is",
    "boxed",
)


@dataclass
class ParseResult:
    parser_id: str
    parsed_option: str | None
    parse_success: bool
    parse_failure_reason: str
    ambiguity_flag: bool
    confidence: float
    span_used: str
    matched_option_text: str
    risk_flags: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "parser_id": self.parser_id,
            "parsed_option": self.parsed_option,
            "parse_success": self.parse_success,
            "parse_failure_reason": self.parse_failure_reason,
            "ambiguity_flag": self.ambiguity_flag,
            "confidence": self.confidence,
            "span_used": self.span_used,
            "matched_option_text": self.matched_option_text,
            "risk_flags": self.risk_flags,
        }


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\\boxed", " boxed ")
    text = text.replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_search(value: Any) -> str:
    text = normalize(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def option_map(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(k).upper().strip(): str(v).strip() for k, v in options.items() if str(k).upper().strip() in OPTION_LETTERS}
    return {}


def letter_regex(letter: str) -> re.Pattern[str]:
    escaped = re.escape(letter.upper())
    return re.compile(
        rf"(?<![A-Za-z0-9])(?:option\s*)?[\(\[\{{]?{escaped}[\)\]\}}\.:\-]?(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def final_answer_span(text: str) -> str:
    norm = normalize(text)
    lower = norm.lower()
    best = -1
    for marker in FINAL_MARKERS:
        idx = lower.rfind(marker)
        if idx > best:
            best = idx
    if best >= 0:
        return norm[best : best + 240]
    return norm[-280:]


def option_letter_mentions(text: str, options: dict[str, str]) -> set[str]:
    return {letter for letter in options if letter_regex(letter).search(text)}


def option_text_mentions(text: str, options: dict[str, str]) -> dict[str, str]:
    haystack = normalize_search(text)
    out: dict[str, str] = {}
    for letter, text_value in options.items():
        needle = normalize_search(text_value)
        if needle and len(needle) >= 4 and needle in haystack:
            out[letter] = text_value
    return out


def fuzzy_text_mentions(text: str, options: dict[str, str], *, threshold: float = 0.92, margin: float = 0.08) -> dict[str, tuple[str, float]]:
    haystack = normalize_search(text)
    scores: list[tuple[str, str, float]] = []
    if not haystack:
        return {}
    for letter, text_value in options.items():
        needle = normalize_search(text_value)
        if not needle or len(needle) < 8:
            continue
        if needle in haystack:
            scores.append((letter, text_value, 1.0))
            continue
        ratio = difflib.SequenceMatcher(None, needle, haystack).ratio()
        scores.append((letter, text_value, ratio))
    scores.sort(key=lambda row: row[2], reverse=True)
    if not scores or scores[0][2] < threshold:
        return {}
    if len(scores) > 1 and scores[0][2] - scores[1][2] < margin:
        return {scores[0][0]: (scores[0][1], scores[0][2]), scores[1][0]: (scores[1][1], scores[1][2])}
    return {scores[0][0]: (scores[0][1], scores[0][2])}


def has_negated_option_ambiguity(text: str) -> bool:
    return bool(re.search(r"\bnot\s+(?:option\s*)?[A-E]\b|\bnot\s+[A-E][\.,;:)]", normalize(text), flags=re.IGNORECASE))


def make_result(
    parser_id: str,
    candidates: set[str],
    span: str,
    confidence: float,
    *,
    matched_option_text: str = "",
    risk_flags: list[str] | None = None,
) -> ParseResult:
    risks = list(risk_flags or [])
    if not candidates:
        return ParseResult(parser_id, None, False, "no_option", False, 0.0, span, matched_option_text, risks)
    if len(candidates) > 1:
        return ParseResult(parser_id, None, False, "ambiguous_multiple_options", True, 0.0, span, matched_option_text, risks + ["multiple_options"])
    letter = next(iter(candidates))
    return ParseResult(parser_id, letter, True, "", False, confidence, span, matched_option_text, risks)


def strict_letter_current(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    parsed = str(prior_parsed or "").upper().strip()
    success = bool(prior_success) and parsed in options
    return ParseResult("strict_letter_current", parsed if success else None, success, "" if success else "strict_no_parse", False, 1.0 if success else 0.0, "", "", [])


def normalized_letter_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    span = normalize(output)
    final = final_answer_span(span)
    final_mentions = option_letter_mentions(final, options)
    if final_mentions:
        risks = ["negated_option_context"] if has_negated_option_ambiguity(final) else []
        return make_result("normalized_letter_parser", final_mentions, final, 0.90, risk_flags=risks)
    mentions = option_letter_mentions(span, options)
    risks = ["no_final_marker"] if mentions else []
    if has_negated_option_ambiguity(span):
        risks.append("negated_option_context")
    return make_result("normalized_letter_parser", mentions, span[-280:], 0.65, risk_flags=risks)


def final_answer_span_letter_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    span = final_answer_span(output)
    risks = ["negated_option_context"] if has_negated_option_ambiguity(span) else []
    return make_result("final_answer_span_letter_parser", option_letter_mentions(span, options), span, 0.95, risk_flags=risks)


def exact_option_text_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    span = normalize(output)
    final = final_answer_span(span)
    final_mentions = option_text_mentions(final, options)
    if final_mentions:
        return make_result("exact_option_text_parser", set(final_mentions), final, 0.86, matched_option_text=" | ".join(final_mentions.values()))
    mentions = option_text_mentions(span, options)
    return make_result("exact_option_text_parser", set(mentions), span[-280:], 0.64, matched_option_text=" | ".join(mentions.values()), risk_flags=["no_final_marker"] if mentions else [])


def final_answer_span_text_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    span = final_answer_span(output)
    mentions = option_text_mentions(span, options)
    return make_result("final_answer_span_text_parser", set(mentions), span, 0.88, matched_option_text=" | ".join(mentions.values()))


def conservative_fuzzy_option_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    span = final_answer_span(output)
    mentions = fuzzy_text_mentions(span, options)
    if len(mentions) > 1:
        return make_result("conservative_fuzzy_option_parser", set(mentions), span, 0.0, matched_option_text=" | ".join(v[0] for v in mentions.values()), risk_flags=["fuzzy_tie"])
    if mentions:
        letter, (text_value, score) = next(iter(mentions.items()))
        return make_result("conservative_fuzzy_option_parser", {letter}, span, min(0.80, score), matched_option_text=text_value, risk_flags=["fuzzy_match"])
    return ParseResult("conservative_fuzzy_option_parser", None, False, "no_fuzzy_option", False, 0.0, span, "", [])


def mcq_robust_ambiguity_rejecting_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    final = final_answer_span(output)
    final_letters = option_letter_mentions(final, options)
    if final_letters:
        result = make_result("MCQ_robust_ambiguity_rejecting_parser", final_letters, final, 0.96)
        if result.parse_success:
            return result
        return result
    final_text = option_text_mentions(final, options)
    if final_text:
        return make_result("MCQ_robust_ambiguity_rejecting_parser", set(final_text), final, 0.88, matched_option_text=" | ".join(final_text.values()))
    phrase = re.search(r"(?:answer\s+is|the\s+answer\s+is)\s*(?:option\s*)?([A-E])\b", normalize(output), flags=re.IGNORECASE)
    if phrase and phrase.group(1).upper() in options:
        return make_result("MCQ_robust_ambiguity_rejecting_parser", {phrase.group(1).upper()}, phrase.group(0), 0.84)
    all_letters = option_letter_mentions(output, options)
    all_text = option_text_mentions(output, options)
    if len(all_letters) > 1 or len(all_text) > 1:
        return ParseResult("MCQ_robust_ambiguity_rejecting_parser", None, False, "ambiguous_without_final_marker", True, 0.0, normalize(output)[-280:], "", ["ambiguity_rejected"])
    if len(all_letters) == 1:
        risks = ["no_final_marker"]
        if has_negated_option_ambiguity(output):
            risks.append("negated_option_context")
            return ParseResult("MCQ_robust_ambiguity_rejecting_parser", None, False, "negated_option_ambiguity", True, 0.0, normalize(output)[-280:], "", risks)
        return make_result("MCQ_robust_ambiguity_rejecting_parser", all_letters, normalize(output)[-280:], 0.65, risk_flags=risks)
    if len(all_text) == 1:
        return make_result("MCQ_robust_ambiguity_rejecting_parser", set(all_text), normalize(output)[-280:], 0.62, matched_option_text=" | ".join(all_text.values()), risk_flags=["no_final_marker"])
    fuzzy = conservative_fuzzy_option_parser(output, options, prior_parsed, prior_success, source)
    if fuzzy.parse_success and not fuzzy.ambiguity_flag:
        fuzzy.parser_id = "MCQ_robust_ambiguity_rejecting_parser"
        return fuzzy
    return ParseResult("MCQ_robust_ambiguity_rejecting_parser", None, False, "no_unambiguous_option", bool(fuzzy.ambiguity_flag), 0.0, final, "", fuzzy.risk_flags)


def source_specific_mmlu_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    source_l = str(source or "").lower()
    if source_l == "sciq" or source_l == "openbookqa":
        result = normalized_letter_parser(output, options, prior_parsed, prior_success, source)
    else:
        result = mcq_robust_ambiguity_rejecting_parser(output, options, prior_parsed, prior_success, source)
    result.parser_id = "source_specific_mmlu_parser"
    return result


PARSER_IDS = (
    "strict_letter_current",
    "normalized_letter_parser",
    "final_answer_span_letter_parser",
    "exact_option_text_parser",
    "final_answer_span_text_parser",
    "conservative_fuzzy_option_parser",
    "source_specific_mmlu_parser",
    "MCQ_robust_ambiguity_rejecting_parser",
)


def parse_with(parser_id: str, output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    if parser_id == "strict_letter_current":
        return strict_letter_current(output, options, prior_parsed, prior_success, source)
    if parser_id == "normalized_letter_parser":
        return normalized_letter_parser(output, options, prior_parsed, prior_success, source)
    if parser_id == "final_answer_span_letter_parser":
        return final_answer_span_letter_parser(output, options, prior_parsed, prior_success, source)
    if parser_id == "exact_option_text_parser":
        return exact_option_text_parser(output, options, prior_parsed, prior_success, source)
    if parser_id == "final_answer_span_text_parser":
        return final_answer_span_text_parser(output, options, prior_parsed, prior_success, source)
    if parser_id == "conservative_fuzzy_option_parser":
        return conservative_fuzzy_option_parser(output, options, prior_parsed, prior_success, source)
    if parser_id == "source_specific_mmlu_parser":
        return source_specific_mmlu_parser(output, options, prior_parsed, prior_success, source)
    if parser_id == "MCQ_robust_ambiguity_rejecting_parser":
        return mcq_robust_ambiguity_rejecting_parser(output, options, prior_parsed, prior_success, source)
    raise ValueError(f"unknown parser_id: {parser_id}")

