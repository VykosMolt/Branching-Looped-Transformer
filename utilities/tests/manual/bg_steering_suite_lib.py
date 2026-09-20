"""Shared helpers for the BG steering and routing experiment suite.

The suite is intentionally file-oriented and resumable. It uses labels, tests,
and answer keys only in evaluator helpers after generation, never during BG
routing or feature capture.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_steering_suite_2026-05-18"
PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
STARTED_AT = time.time()
MAX_WALL_SECONDS = 8 * 60 * 60
MAX_TOTAL_TASKS = 80
MAX_TOTAL_BRANCHES = 1000
SEED = 20260518

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def out_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_json(path: str | Path, default: Any = None) -> Any:
    p = out_path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)


def write_md(path: str | Path, lines: Sequence[str]) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def snippet(text: Any, limit: int = 220) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def now_elapsed() -> float:
    return time.time() - STARTED_AT


def wall_time_exceeded() -> bool:
    return now_elapsed() > MAX_WALL_SECONDS


def ensure_report_root() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def options_text(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def mcq_prompt(question: str, options: dict[str, str]) -> str:
    return (
        "Answer the multiple-choice question. Give concise reasoning if useful, "
        "then end with exactly one line: FINAL ANSWER: <letter>.\n\n"
        f"Question:\n{question}\n\nOptions:\n{options_text(options)}"
    )


def gsm_prompt(question: str) -> str:
    return (
        "Solve the arithmetic word problem concisely. End with exactly one line: "
        "FINAL ANSWER: <number>.\n\n"
        f"Problem:\n{question}"
    )


def task_generation_prompt(task: dict[str, Any]) -> str:
    domain = task["domain"]
    if domain == "code":
        return (
            str(task["prompt"]).strip()
            + "\n\nProduce the best solution you can within the token budget. "
            + "Complete code is preferred, but an incomplete partial implementation is allowed."
        )
    if domain in {"reasoning", "science"}:
        return mcq_prompt(task["question"], task["options"])
    if domain == "gsm8k":
        return gsm_prompt(task["question"])
    return str(task.get("prompt") or task.get("question") or "")


def continuation_prompt(task: dict[str, Any], partial_text: str, max_tokens: int | None = None) -> str:
    del max_tokens
    domain = task["domain"]
    if domain == "code":
        return (
            str(task["prompt"]).strip()
            + "\n\nPartial solution so far:\n"
            + str(partial_text).strip()
            + "\n\nContinue from the partial solution and finish with complete Python code only."
        )
    if domain in {"reasoning", "science"}:
        return (
            mcq_prompt(task["question"], task["options"])
            + "\n\nPartial answer attempt:\n"
            + str(partial_text).strip()
            + "\n\nContinue and end with FINAL ANSWER: <letter>."
        )
    if domain == "gsm8k":
        return (
            gsm_prompt(task["question"])
            + "\n\nPartial solution so far:\n"
            + str(partial_text).strip()
            + "\n\nContinue and end with FINAL ANSWER: <number>."
        )
    return str(task.get("prompt") or "") + "\n\n" + str(partial_text)


def partial_budget(task: dict[str, Any]) -> int:
    if task["domain"] == "code" and task.get("is_devil"):
        return 256
    if task["domain"] == "code":
        return 192
    if task["domain"] in {"reasoning", "science"}:
        return 96
    if task["domain"] == "gsm8k":
        return 128
    return 96


def continuation_budget(task: dict[str, Any]) -> int:
    if task["domain"] == "code" and task.get("is_devil"):
        return 768
    if task["domain"] == "code":
        return 512
    if task["domain"] in {"reasoning", "science"}:
        return 192
    if task["domain"] == "gsm8k":
        return 256
    return 192


def equal_budget_per_branch(task: dict[str, Any]) -> int:
    if task["domain"] == "code":
        return 256
    if task["domain"] in {"reasoning", "science"}:
        return 96
    if task["domain"] == "gsm8k":
        return 128
    return 96


def normalize_number(text: str) -> Fraction | None:
    s = str(text).strip()
    s = s.replace(",", "")
    s = s.strip("$% \t\n.:")
    if not s:
        return None
    try:
        return Fraction(s)
    except Exception:
        pass
    try:
        return Fraction(float(s)).limit_denominator(1_000_000)
    except Exception:
        return None


def parse_gsm_answer(text: str) -> str | None:
    raw = str(text or "")
    patterns = [
        r"FINAL ANSWER\s*:\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)",
        r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)",
        r"answer\s*(?:is|:)\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)",
    ]
    for pattern in patterns:
        found = re.findall(pattern, raw, flags=re.IGNORECASE)
        if found:
            return found[-1].strip()
    found = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", raw)
    return found[-1].strip() if found else None


def parse_mcq_answer(text: str, options: dict[str, str]) -> str | None:
    raw = str(text or "")
    letters = sorted(str(k).upper() for k in options)
    allowed = "".join(re.escape(x) for x in letters)
    patterns = [
        rf"FINAL ANSWER\s*:\s*<?([{allowed}])>?",
        rf"\banswer\s*(?:is|:)\s*<?([{allowed}])>?",
        rf"\b([{allowed}])\s*[\).]\s",
    ]
    for pattern in patterns:
        found = re.findall(pattern, raw, flags=re.IGNORECASE)
        if found:
            return str(found[-1]).upper()
    compact = raw.strip().upper()
    if compact in letters:
        return compact
    return None


_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str:
    raw = str(text or "").strip()
    match = _PY_FENCE_RE.search(raw) or _ANY_FENCE_RE.search(raw)
    if match:
        raw = match.group(1).strip()
    if "FINAL ANSWER:" in raw.upper():
        raw = re.split(r"FINAL ANSWER\s*:", raw, flags=re.IGNORECASE)[-1]
        match = _PY_FENCE_RE.search(raw) or _ANY_FENCE_RE.search(raw)
        if match:
            raw = match.group(1).strip()
    idx = raw.find("def ")
    if idx >= 0:
        raw = raw[idx:]
    stop_markers = ("[Observation]", "[Verifier]", "[System]", "<END_FINAL>")
    for marker in stop_markers:
        if marker in raw:
            raw = raw.split(marker, 1)[0]
    return raw.strip()


SAFE_IMPORT_ROOTS = {
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "typing",
}
DANGEROUS_IMPORT_ROOTS = {"ctypes", "multiprocessing", "os", "pathlib", "requests", "shutil", "signal", "socket", "subprocess", "urllib"}
DANGEROUS_CALL_NAMES = {"__import__", "compile", "eval", "exec", "input", "open"}
SAFE_SYS_IMPORT_NAMES = {"setrecursionlimit"}


def _root_name(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def safety_check(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return True, f"syntax_deferred:{exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root == "sys":
                    continue
                if root in DANGEROUS_IMPORT_ROOTS or root not in SAFE_IMPORT_ROOTS:
                    return False, f"unsafe import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root == "sys" and all(alias.name in SAFE_SYS_IMPORT_NAMES for alias in node.names):
                continue
            if root in DANGEROUS_IMPORT_ROOTS or root not in SAFE_IMPORT_ROOTS:
                return False, f"unsafe import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALL_NAMES:
                return False, f"unsafe call: {node.func.id}"
            if isinstance(node.func, ast.Attribute):
                root = _root_name(node.func.value)
                if root == "sys" and node.func.attr in SAFE_SYS_IMPORT_NAMES:
                    continue
                if root in DANGEROUS_IMPORT_ROOTS:
                    return False, f"unsafe attribute call: {root}.{node.func.attr}"
    return True, ""


def eval_python_candidate(code: str, tests: list[str], function_name: str, timeout: float = 5.0) -> dict[str, Any]:
    safety_ok, safety_reason = safety_check(code)
    if not safety_ok:
        return {
            "success": False,
            "syntax_ok": False,
            "runtime_ok": False,
            "safety_ok": False,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "error_type": "SafetyRejected",
            "error_message": safety_reason,
        }
    script = r"""
import contextlib
import io
import json
import sys
import time
import traceback
payload=json.loads(sys.stdin.read())
code=payload["code"]
tests=payload["tests"]
function_name=payload["function_name"]
scope={"__name__":"__candidate_eval__"}
stdout=io.StringIO()
stderr=io.StringIO()
result={"syntax_ok":False,"runtime_ok":False,"safety_ok":True,"tests_total":len(tests),"tests_passed":0,"tests_failed":len(tests),"error_type":"","error_message":"","stdout":"","stderr":""}
started=time.perf_counter()
try:
    compiled=compile(code, "<candidate>", "exec")
    result["syntax_ok"]=True
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exec(compiled, scope, scope)
    if function_name and not callable(scope.get(function_name)):
        raise AssertionError("function not found: " + function_name)
    for idx,test in enumerate(tests):
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(test, scope, scope)
            result["tests_passed"] += 1
        except Exception as exc:
            if not result["error_type"]:
                result["error_type"] = type(exc).__name__
                result["error_message"] = str(exc)[:500]
    result["tests_failed"] = len(tests) - result["tests_passed"]
    result["runtime_ok"] = True
except Exception as exc:
    result["error_type"] = type(exc).__name__
    result["error_message"] = str(exc)[:500]
result["seconds"] = time.perf_counter() - started
result["stdout"] = stdout.getvalue()[-500:]
result["stderr"] = stderr.getvalue()[-500:]
result["success"] = result["tests_total"] > 0 and result["tests_passed"] == result["tests_total"]
print("JSON_RESULT:" + json.dumps(result))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps({"code": code, "tests": tests, "function_name": function_name}),
            text=True,
            capture_output=True,
            timeout=max(0.5, timeout + 1.0),
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "syntax_ok": False,
            "runtime_ok": False,
            "safety_ok": True,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "error_type": "TimeoutExpired",
            "error_message": "candidate timed out",
        }
    parsed = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("JSON_RESULT:"):
            parsed = json.loads(line[len("JSON_RESULT:"):])
            break
    if parsed is None:
        parsed = {
            "success": False,
            "syntax_ok": False,
            "runtime_ok": False,
            "safety_ok": True,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "error_type": "HarnessParseError",
            "error_message": (proc.stderr or proc.stdout)[-500:],
        }
    parsed["pass_rate"] = float(parsed.get("tests_passed", 0)) / max(int(parsed.get("tests_total", 0)), 1)
    return parsed


def evaluate_output(task: dict[str, Any], text: str) -> dict[str, Any]:
    domain = task["domain"]
    if domain == "code":
        code = extract_python_code(text)
        tests = list(task.get("public_tests") or []) + list(task.get("hidden_tests") or []) + list(task.get("tests") or [])
        result = eval_python_candidate(code, tests, str(task.get("function_name") or ""), float(task.get("timeout_seconds") or 5.0))
        result.update({"evaluable": bool(tests), "parsed": bool(code.strip()), "extracted_code_chars": len(code)})
        return result
    if domain in {"reasoning", "science"}:
        parsed = parse_mcq_answer(text, task["options"])
        success = parsed == str(task.get("answer_key") or task.get("answer")).upper()
        return {"evaluable": True, "parsed": parsed is not None, "parsed_answer": parsed, "success": bool(success)}
    if domain == "gsm8k":
        parsed = parse_gsm_answer(text)
        p = normalize_number(parsed or "")
        g = normalize_number(str(task.get("answer_key") or task.get("gold_answer") or ""))
        success = p is not None and g is not None and p == g
        return {"evaluable": True, "parsed": parsed is not None, "parsed_answer": parsed, "success": bool(success)}
    return {"evaluable": False, "success": False, "error": "unknown domain"}


class OuroTextGenerator:
    """Thin direct local Ouro generator using the already implemented extractor loader."""

    def __init__(self, device: str = "cuda", dtype: str = "auto") -> None:
        from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

        self.extractor = BGTransformerFeatureExtractor(device=device, dtype=dtype, force_all_loops=True)
        self.model = self.extractor.model
        self.tokenizer = self.extractor.tokenizer
        self.device = self.extractor.device

    def cleanup(self) -> None:
        self.extractor.cleanup()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = SEED,
    ) -> dict[str, Any]:
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        prompt_len = int(enc["input_ids"].shape[1])
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        started = time.time()
        hit_error = ""
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            hit_error = "cuda_oom_retry_no_cache"
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            with torch.inference_mode():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=max(1, max_new_tokens // 2),
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )
        new_ids = generated[0, prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        full = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        if not text and full.startswith(prompt):
            text = full[len(prompt):].strip()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "text": text,
            "raw_text": text,
            "token_count": int(new_ids.numel()),
            "hit_max_tokens": int(new_ids.numel()) >= max_new_tokens,
            "generation_error": hit_error,
            "seconds": round(time.time() - started, 3),
        }


def load_code_tasks() -> list[dict[str, Any]]:
    paths = [
        PROBE_ROOT / "code_branch_taskset_v2_near_miss10_2026-05-17.json",
        PROBE_ROOT / "code_branch_taskset_v2_mini_patched_2026-05-16.json",
        PROBE_ROOT / "bg_devil_task_offline_dynamic_connectivity_2026-05-18.json",
        PROBE_ROOT / "bg_devil_task_minimum_xor_paths_2026-05-18.json",
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        payload = load_json(path, {})
        candidates = payload.get("tasks") if isinstance(payload, dict) else None
        if candidates is None and isinstance(payload, dict) and payload.get("task_id"):
            candidates = [payload]
        for item in candidates or []:
            task_id = str(item.get("task_id") or item.get("function_name") or "")
            if not task_id or task_id in seen:
                continue
            seen.add(task_id)
            row = dict(item)
            row.update(
                {
                    "domain": "code",
                    "source_dataset": row.get("source") or row.get("source_dataset") or "code_local",
                    "answer_key": None,
                    "is_devil": "offline_dynamic_connectivity" in task_id or "minimum_xor_paths" in task_id or row.get("difficulty") == "devil",
                    "available_wrapper_mode": False,
                }
            )
            rows.append(row)
    return rows


def load_reasoning_tasks() -> list[dict[str, Any]]:
    payload = load_json(PROBE_ROOT / "reasoning_branch_pilot_2026-05-17.json", {})
    rows = []
    for item in list(payload.get("tasks") or []):
        answer = str(item.get("answer") or item.get("answer_key") or "").upper()
        if not answer or not item.get("options"):
            continue
        rows.append(
            {
                "task_id": str(item["task_id"]),
                "domain": "reasoning",
                "source_dataset": item.get("dataset", "reasoning"),
                "question": item["question"],
                "options": item["options"],
                "answer_key": answer,
                "prompt": mcq_prompt(item["question"], item["options"]),
                "difficulty": "mixed",
                "is_devil": False,
                "available_wrapper_mode": False,
            }
        )
    return rows


def load_science_tasks() -> list[dict[str, Any]]:
    payload = load_json(PROBE_ROOT / "science_natural_distractor_set_2026-05-17.json", {})
    rows = []
    for item in list(payload.get("tasks") or []):
        answer = str(item.get("answer_key") or item.get("answer") or "").upper()
        if not answer or not item.get("options"):
            continue
        rows.append(
            {
                "task_id": str(item["task_id"]),
                "domain": "science",
                "source_dataset": item.get("source_dataset", "science"),
                "source_subject": item.get("source_subject"),
                "subdomain_bucket": item.get("subdomain_bucket"),
                "question": item["question"],
                "options": item["options"],
                "answer_key": answer,
                "prompt": mcq_prompt(item["question"], item["options"]),
                "difficulty": "mixed",
                "is_devil": False,
                "available_wrapper_mode": False,
            }
        )
    return rows


def load_gsm8k_tasks() -> list[dict[str, Any]]:
    payload = load_json(PROBE_ROOT / "clean_gsm8k_expanded_2026-05-16.json", {})
    rows = []
    seen: set[str] = set()
    for item in list(payload.get("tournaments") or []):
        task_id = f"gsm8k/{item.get('dataset_index', item.get('problem_id'))}"
        if task_id in seen:
            continue
        seen.add(task_id)
        question = str(item.get("question") or "")
        answer = str(item.get("gold_answer") or "")
        if not question or not answer:
            continue
        rows.append(
            {
                "task_id": task_id,
                "domain": "gsm8k",
                "source_dataset": "gsm8k",
                "question": question,
                "answer_key": answer,
                "gold_answer": answer,
                "prompt": gsm_prompt(question),
                "difficulty": "clean",
                "is_devil": False,
                "available_wrapper_mode": False,
            }
        )
    for item in list(payload.get("problem_diagnostics") or []):
        if len(rows) >= 30:
            break
        task_id = f"gsm8k/{item.get('dataset_index')}"
        if task_id in seen:
            continue
        seen.add(task_id)
        question = str(item.get("question") or "")
        answer = str(item.get("gold_answer") or "")
        if question and answer:
            rows.append(
                {
                    "task_id": task_id,
                    "domain": "gsm8k",
                    "source_dataset": "gsm8k",
                    "question": question,
                    "answer_key": answer,
                    "gold_answer": answer,
                    "prompt": gsm_prompt(question),
                    "difficulty": "clean",
                    "is_devil": False,
                    "available_wrapper_mode": False,
                }
            )
    return rows


def build_task_suite_rows() -> list[dict[str, Any]]:
    code_all = load_code_tasks()
    devil = [t for t in code_all if t.get("is_devil")]
    hard = [t for t in code_all if not t.get("is_devil") and str(t.get("difficulty", "")).lower() in {"hard", "hard_not_devil", "devil"}]
    medium = [t for t in code_all if not t.get("is_devil") and t not in hard]
    code = (medium[:8] + hard[:5] + devil[:2] + medium[8:13] + hard[5:10])[:20]
    reasoning = load_reasoning_tasks()[:15]
    science = load_science_tasks()[:15]
    gsm = load_gsm8k_tasks()[:10]
    rows = code + reasoning + science + gsm
    rows = rows[:MAX_TOTAL_TASKS]
    for idx, row in enumerate(rows):
        row["suite_index"] = idx
    return rows


def load_task_suite() -> list[dict[str, Any]]:
    payload = load_json(REPORT_ROOT / "task_suite.json", {})
    return list(payload.get("tasks") or [])


def task_by_id(tasks: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(task["task_id"]): task for task in tasks}


def load_branch_pools() -> dict[str, Any]:
    return load_json(REPORT_ROOT / "branch_pools.json", load_json(REPORT_ROOT / "branch_pools.partial.json", {}))


def branch_rows_for_task(branch_payload: dict[str, Any], task_id: str) -> list[dict[str, Any]]:
    rows = []
    for row in branch_payload.get("branches") or []:
        if str(row.get("task_id")) == str(task_id):
            rows.append(row)
    return sorted(rows, key=lambda r: int(r.get("branch_id", 0)))


def domain_hint_for_task(task: dict[str, Any]) -> str:
    if task["domain"] == "code":
        return "code"
    if task["domain"] == "reasoning":
        return "reasoning"
    if task["domain"] == "science":
        return "science"
    if task["domain"] == "gsm8k":
        return "gsm8k"
    return "objective"


def summarize_success(values: Iterable[bool]) -> dict[str, Any]:
    vals = [bool(v) for v in values]
    return {"n": len(vals), "successes": sum(vals), "rate": sum(vals) / max(len(vals), 1)}


def bootstrap_delta(a: list[float], b: list[float], seed: int = SEED, rounds: int = 500) -> dict[str, float]:
    if not a or len(a) != len(b):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    deltas = []
    n = len(a)
    for _ in range(rounds):
        idxs = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in idxs) / n)
    deltas.sort()
    return {"mean": sum(deltas) / len(deltas), "lo": deltas[int(0.025 * rounds)], "hi": deltas[int(0.975 * rounds)]}


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            rank = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = rank
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    deny = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (denx * deny) if denx > 0 and deny > 0 else float("nan")


def auc_score(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
            total += 1
    return wins / max(total, 1)


def verdict_from_delta(delta: float, n: int, min_n: int = 20) -> str:
    if n < min_n:
        return "INSUFFICIENT"
    if delta >= 0.05:
        return "HELPS"
    if delta < -0.05:
        return "HURTS"
    return "NEUTRAL"


def load_reachability() -> dict[str, Any]:
    return load_json(REPORT_ROOT / "reachability_gate.json", {})


def domains_allowed_by_reachability() -> set[str]:
    payload = load_reachability()
    verdict = payload.get("BG_REACHABILITY_GATE_VERDICT") or payload.get("verdict")
    if verdict == "READY":
        return {"code", "reasoning", "science", "gsm8k"}
    if verdict == "PARTIAL":
        by_domain = payload.get("by_domain") or {}
        return {d for d, row in by_domain.items() if row.get("meets_threshold")}
    return set()


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return out
