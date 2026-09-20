"""Experimental science MCQ parser candidates for repair v2.

This module is local to manual probes. It does not replace the production or
historical strict parser. Callers must explicitly choose a parser candidate.
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
    "so the answer is",
    "therefore",
    "thus",
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
            "risk_flags": self.risk_flags,
        }


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\\boxed", " boxed ")
    text = text.replace("＂", '"').replace("`", "'")
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
    letter = re.escape(letter.upper())
    return re.compile(
        rf"(?<![A-Za-z0-9])(?:option\s*)?[\(\[\{{]?{letter}[\)\]\}}\.:\-]?(?![A-Za-z0-9])",
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
        return norm[best : best + 220]
    return norm[-260:]


def letter_mentions(text: str, options: dict[str, str]) -> set[str]:
    return {letter for letter in options if letter_regex(letter).search(text)}


def exact_text_mentions(text: str, options: dict[str, str]) -> set[str]:
    haystack = normalize_search(text)
    out = set()
    for letter, option_text in options.items():
        needle = normalize_search(option_text)
        if needle and len(needle) >= 4 and needle in haystack:
            out.add(letter)
    return out


def fuzzy_text_mentions(text: str, options: dict[str, str], *, threshold: float = 0.94) -> set[str]:
    haystack = normalize_search(text)
    out = set()
    if not haystack:
        return out
    for letter, option_text in options.items():
        needle = normalize_search(option_text)
        if not needle or len(needle) < 8:
            continue
        if needle in haystack:
            out.add(letter)
            continue
        # Sliding matching is expensive and noisy; use a conservative whole-text ratio.
        ratio = difflib.SequenceMatcher(None, needle, haystack).quick_ratio()
        if ratio >= threshold:
            out.add(letter)
    return out


def choose_single(parser_id: str, candidates: set[str], span: str, confidence: float, risk_flags: list[str] | None = None) -> ParseResult:
    risk = list(risk_flags or [])
    if not candidates:
        return ParseResult(parser_id, None, False, "no_option", False, 0.0, span, risk)
    if len(candidates) > 1:
        return ParseResult(parser_id, None, False, "ambiguous_multiple_options", True, 0.0, span, risk + ["multiple_options"])
    value = next(iter(candidates))
    return ParseResult(parser_id, value, True, "", False, confidence, span, risk)


def strict_letter_current(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None) -> ParseResult:
    parsed = str(prior_parsed or "").upper().strip()
    success = bool(prior_success) and parsed in options
    return ParseResult("strict_letter_current", parsed if success else None, success, "" if success else "strict_no_parse", False, 1.0 if success else 0.0, "", [])


def normalized_letter_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None) -> ParseResult:
    span = normalize(output)
    final = final_answer_span(span)
    final_mentions = letter_mentions(final, options)
    if final_mentions:
        return choose_single("normalized_letter_parser", final_mentions, final, 0.92)
    all_mentions = letter_mentions(span, options)
    return choose_single("normalized_letter_parser", all_mentions, span[-260:], 0.70, ["no_final_marker"] if all_mentions else [])


def option_text_exact_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None) -> ParseResult:
    span = normalize(output)
    final = final_answer_span(span)
    final_mentions = exact_text_mentions(final, options)
    if final_mentions:
        return choose_single("option_text_exact_parser", final_mentions, final, 0.86)
    mentions = exact_text_mentions(span, options)
    return choose_single("option_text_exact_parser", mentions, span[-260:], 0.68, ["no_final_marker"] if mentions else [])


def option_text_fuzzy_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None) -> ParseResult:
    span = normalize(output)
    final = final_answer_span(span)
    final_mentions = fuzzy_text_mentions(final, options)
    if final_mentions:
        return choose_single("option_text_fuzzy_parser", final_mentions, final, 0.72, ["fuzzy_match"])
    mentions = fuzzy_text_mentions(span, options)
    return choose_single("option_text_fuzzy_parser", mentions, span[-260:], 0.55, ["fuzzy_match", "no_final_marker"] if mentions else [])


def rationale_then_final_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None) -> ParseResult:
    span = final_answer_span(output)
    letters = letter_mentions(span, options)
    if letters:
        return choose_single("rationale_then_final_parser", letters, span, 0.94)
    exact = exact_text_mentions(span, options)
    if exact:
        return choose_single("rationale_then_final_parser", exact, span, 0.86)
    return ParseResult("rationale_then_final_parser", None, False, "no_final_answer_option", False, 0.0, span, [])


def mcq_robust_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None) -> ParseResult:
    strict = strict_letter_current(output, options, prior_parsed, prior_success)
    if strict.parse_success:
        strict.parser_id = "MCQ_robust_parser"
        return strict
    final = rationale_then_final_parser(output, options, prior_parsed, prior_success)
    if final.parse_success:
        final.parser_id = "MCQ_robust_parser"
        return final
    exact = option_text_exact_parser(output, options, prior_parsed, prior_success)
    if exact.parse_success and not exact.ambiguity_flag:
        exact.parser_id = "MCQ_robust_parser"
        return exact
    fuzzy = option_text_fuzzy_parser(output, options, prior_parsed, prior_success)
    if fuzzy.parse_success and not fuzzy.ambiguity_flag and fuzzy.confidence >= 0.72:
        fuzzy.parser_id = "MCQ_robust_parser"
        return fuzzy
    risks = []
    if exact.ambiguity_flag or fuzzy.ambiguity_flag:
        risks.append("ambiguity_rejected")
    return ParseResult("MCQ_robust_parser", None, False, "no_unambiguous_option", bool(risks), 0.0, final_answer_span(output), risks)


def source_specific_parser(output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    # SciQ generations often include boxed letters; MMLU benefits from final-span
    # handling and exact option text, but fuzzy matching stays diagnostic only.
    source_l = str(source or "").lower()
    if source_l == "sciq":
        result = normalized_letter_parser(output, options, prior_parsed, prior_success)
    else:
        result = mcq_robust_parser(output, options, prior_parsed, prior_success)
    result.parser_id = "source_specific_parser"
    return result


PARSER_IDS = (
    "strict_letter_current",
    "normalized_letter_parser",
    "option_text_exact_parser",
    "option_text_fuzzy_parser",
    "rationale_then_final_parser",
    "MCQ_robust_parser",
    "source_specific_parser",
)


def parse_with(parser_id: str, output: str, options: dict[str, str], prior_parsed: Any = None, prior_success: Any = None, source: str = "") -> ParseResult:
    if parser_id == "strict_letter_current":
        return strict_letter_current(output, options, prior_parsed, prior_success)
    if parser_id == "normalized_letter_parser":
        return normalized_letter_parser(output, options, prior_parsed, prior_success)
    if parser_id == "option_text_exact_parser":
        return option_text_exact_parser(output, options, prior_parsed, prior_success)
    if parser_id == "option_text_fuzzy_parser":
        return option_text_fuzzy_parser(output, options, prior_parsed, prior_success)
    if parser_id == "rationale_then_final_parser":
        return rationale_then_final_parser(output, options, prior_parsed, prior_success)
    if parser_id == "MCQ_robust_parser":
        return mcq_robust_parser(output, options, prior_parsed, prior_success)
    if parser_id == "source_specific_parser":
        return source_specific_parser(output, options, prior_parsed, prior_success, source)
    raise ValueError(f"unknown parser_id: {parser_id}")

