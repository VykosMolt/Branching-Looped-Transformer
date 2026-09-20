"""Run a bounded live Ouro-RLTT BG best-of-N smoke test."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.bg_controller import BGController  # noqa: E402
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "opi/taps/probes"
AGG_JSON = REPORT_DIR / "bg_transformer_best_of_n_smoke_all_2026-05-18.json"
AGG_MD = REPORT_DIR / "bg_transformer_best_of_n_smoke_all_2026-05-18.md"
SUPPORTED_DOMAIN_HINTS = (
    "hh",
    "preference",
    "unknown",
    "code",
    "strict_clean_code",
    "reasoning",
    "science",
    "math",
    "gsm8k",
    "objective",
)
SUPPORTED_MODES = ("conservative", "experimental_vote", "code_backup", "diagnostic_all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--domain-hint", choices=SUPPORTED_DOMAIN_HINTS, default="objective")
    parser.add_argument("--n-candidates", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="conservative")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=str(REPORT_DIR / "bg_transformer_best_of_n_smoke_2026-05-18.json"))
    parser.add_argument("--optional-test-json", default="")
    parser.add_argument("--feature-max-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    raise ValueError("provide --prompt or --prompt-file")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _device_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(idx)
        info.update(
            {
                "cuda_device_index": idx,
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_free_bytes": int(free),
                "cuda_total_bytes": int(total),
            }
        )
    return info


def _clean_candidate_text(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^\s*(assistant|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _generate_candidates(
    extractor: BGTransformerFeatureExtractor,
    prompt: str,
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict[str, Any]]:
    tokenizer = extractor.tokenizer
    model = extractor.model
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(extractor.device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    out: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for idx in range(n_candidates):
        torch.manual_seed(seed + idx)
        if extractor.device.type == "cuda":
            torch.cuda.manual_seed_all(seed + idx)
        with torch.inference_mode():
            try:
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if extractor.device.type == "cuda":
                    torch.cuda.empty_cache()
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
        new_ids = generated[0, prompt_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if not text.strip():
            full = tokenizer.decode(generated[0], skip_special_tokens=True)
            text = full[len(prompt):] if full.startswith(prompt) else full
        out.append(
            {
                "index": idx,
                "text": _clean_candidate_text(text),
                "token_count": int(new_ids.numel()),
            }
        )
        if extractor.device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _extract_python_code(candidate_text: str) -> str:
    matches = re.findall(r"```(?:python|py)?\s*(.*?)```", candidate_text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n\n".join(m.strip() for m in matches if m.strip()).strip()
    idx = candidate_text.find("def ")
    if idx >= 0:
        return candidate_text[idx:].strip()
    return candidate_text.strip()


def _run_candidate_tests(candidate_text: str, test_spec: dict[str, Any]) -> dict[str, Any]:
    tests = list(test_spec.get("public_tests") or []) + list(test_spec.get("hidden_tests") or []) + list(test_spec.get("tests") or [])
    timeout = float(test_spec.get("timeout_seconds") or 8.0)
    if not tests:
        return {"status": "TESTS_SKIPPED", "reason": "no tests in optional test JSON"}
    runner = r"""
import json
import sys
import traceback

payload = json.load(sys.stdin)
namespace = {}
result = {"status": "PASS", "passed": 0, "failed": 0, "failures": []}
try:
    exec(payload["code"], namespace)
except Exception:
    result["status"] = "FAIL"
    result["failed"] = len(payload["tests"])
    result["failures"].append({"stage": "exec_candidate", "traceback": traceback.format_exc()})
else:
    for idx, test in enumerate(payload["tests"]):
        try:
            exec(test, namespace)
            result["passed"] += 1
        except Exception:
            result["status"] = "FAIL"
            result["failed"] += 1
            result["failures"].append({"test_index": idx, "traceback": traceback.format_exc()})
print(json.dumps(result))
"""
    code = _extract_python_code(candidate_text)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            input=json.dumps({"code": code, "tests": tests}),
            text=True,
            capture_output=True,
            timeout=timeout + 2.0,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "reason": "timeout", "timeout_seconds": timeout}
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        parsed = {"status": "FAIL", "reason": "unparseable_test_runner_output", "stdout": proc.stdout, "stderr": proc.stderr}
    parsed["returncode"] = proc.returncode
    parsed["candidate_code_chars"] = len(code)
    parsed["test_count"] = len(tests)
    if proc.stderr.strip():
        parsed["stderr"] = proc.stderr[-2000:]
    return parsed


def _run_optional_tests(candidates: list[dict[str, Any]], optional_test_json: str) -> dict[str, Any] | None:
    if not optional_test_json:
        return None
    path = Path(optional_test_json)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {"status": "TESTS_SKIPPED", "reason": f"optional test JSON not found: {path}"}
    spec = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "RAN",
        "source_path": str(path),
        "task_id": spec.get("task_id"),
        "function_name": spec.get("function_name"),
        "tests_visibility": spec.get("tests_visibility"),
        "candidate_results": [
            {"index": row["index"], **_run_candidate_tests(str(row.get("text", "")), spec)}
            for row in candidates
        ],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# BG Transformer Best-of-N Smoke: {path.stem}",
        "",
        f"BG_TRANSFORMER_BEST_OF_N_SMOKE_VERDICT = {payload.get('verdict')}",
        "",
        "## Selection",
        "",
        f"- Domain hint: `{payload.get('domain_hint')}`",
        f"- Mode: `{payload.get('mode')}`",
        f"- Selected index: `{payload.get('selected_index')}`",
        f"- Feature shapes: `{payload.get('feature_shapes')}`",
        "",
        "## Candidates",
        "",
    ]
    for candidate in payload.get("candidates", []):
        selected = " selected" if candidate.get("index") == payload.get("selected_index") else ""
        text = str(candidate.get("text", "")).strip()
        if len(text) > 1600:
            text = text[:1600] + "\n...[truncated]"
        lines.extend(
            [
                f"### Candidate {candidate.get('index')}{selected}",
                "",
                f"- Token count: `{candidate.get('token_count')}`",
                "",
                "```text",
                text,
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Controller Result",
            "",
            "```json",
            json.dumps(_jsonable(payload.get("selection_result")), indent=2, sort_keys=True)[:8000],
            "```",
        ]
    )
    if payload.get("optional_test_results"):
        lines.extend(
            [
                "",
                "## Optional Test Results",
                "",
                "```json",
                json.dumps(_jsonable(payload["optional_test_results"]), indent=2, sort_keys=True)[:8000],
                "```",
            ]
        )
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in payload["warnings"])
    if payload.get("error"):
        lines.extend(["", "## Error", "", "```text", str(payload["error"])[-6000:], "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload, output.with_suffix(".md"))
    _write_aggregate_report()


def _is_devil_payload(payload: dict[str, Any], path: Path) -> bool:
    task_id = str((payload.get("optional_test_results") or {}).get("task_id", ""))
    name = path.stem
    return "offline_dynamic_connectivity" in task_id or "minimum_xor_paths" in task_id or "offline_dynamic_connectivity" in name or "minimum_xor_paths" in name


def _write_aggregate_report() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(REPORT_DIR.glob("bg_transformer_best_of_n_smoke_*_2026-05-18.json")):
        if path.name == AGG_JSON.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "verdict" not in payload:
            continue
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "name": path.stem,
                "domain_hint": payload.get("domain_hint"),
                "mode": payload.get("mode"),
                "verdict": payload.get("verdict"),
                "selected_index": payload.get("selected_index"),
                "candidate_count": len(payload.get("candidates") or []),
                "feature_shapes": payload.get("feature_shapes"),
                "is_devil": _is_devil_payload(payload, path),
            }
        )
    non_devil_pass = sum(1 for row in rows if not row["is_devil"] and row["verdict"] == "PASS")
    non_devil_any = sum(1 for row in rows if not row["is_devil"] and row["verdict"] in {"PASS", "PARTIAL"})
    if non_devil_pass >= 3:
        integration_verdict = "PASS"
    elif non_devil_any >= 1:
        integration_verdict = "PARTIAL"
    else:
        integration_verdict = "FAIL"
    devil_rows = [row for row in rows if row["is_devil"]]
    if len(devil_rows) >= 2 and all(row["verdict"] in {"PASS", "PARTIAL"} for row in devil_rows):
        devil_verdict = "PASS"
    elif len(devil_rows) >= 1 and any(row["verdict"] in {"PASS", "PARTIAL"} for row in devil_rows):
        devil_verdict = "PARTIAL"
    elif len(devil_rows) == 0:
        devil_verdict = "SKIPPED"
    else:
        devil_verdict = "FAIL"

    payload = {
        "BG_TRANSFORMER_INTEGRATION_VERDICT": integration_verdict,
        "BG_DEVIL_BEST_OF_N_VERDICT": devil_verdict,
        "smoke_count": len(rows),
        "smokes": rows,
        "updated_by": Path(__file__).name,
    }
    AGG_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# BG Transformer Best-of-N Smoke Aggregate (2026-05-18)",
        "",
        f"BG_TRANSFORMER_INTEGRATION_VERDICT = {integration_verdict}",
        f"BG_DEVIL_BEST_OF_N_VERDICT = {devil_verdict}",
        "",
        "| Smoke | Domain | Verdict | Selected | Candidates | Devil |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['domain_hint']}` | `{row['verdict']}` | "
            f"`{row['selected_index']}` | `{row['candidate_count']}` | `{row['is_devil']}` |"
        )
    AGG_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_once(args: argparse.Namespace, prompt: str, warnings: list[str]) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t0 = time.time()
    with BGTransformerFeatureExtractor(device=args.device, dtype="auto", force_all_loops=True) as extractor:
        timings["load_seconds"] = round(time.time() - t0, 3)

        t1 = time.time()
        candidates = _generate_candidates(
            extractor,
            prompt,
            args.n_candidates,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.seed,
        )
        timings["generation_seconds"] = round(time.time() - t1, 3)
        if len(candidates) < 2:
            warnings.append("fewer than two candidates were generated")

        t2 = time.time()
        feature_rows = [
            extractor.encode_prompt_candidate(
                prompt,
                str(candidate["text"]),
                domain_hint=args.domain_hint,
                max_length=args.feature_max_length,
            )
            for candidate in candidates
        ]
        features = torch.stack(feature_rows, dim=0)
        timings["feature_capture_seconds"] = round(time.time() - t2, 3)

    t3 = time.time()
    controller = BGController.from_artifacts(device="cpu")
    selection_result = controller.select_best(
        features,
        domain_hint=args.domain_hint,
        mode=args.mode,
        return_details=True,
    )
    conservative_result = controller.select_best(
        features,
        domain_hint=args.domain_hint,
        mode="conservative",
        return_details=True,
    )
    diagnostic_all = controller.select_best(
        features,
        domain_hint=args.domain_hint,
        mode="diagnostic_all",
        return_details=True,
    )
    experimental_vote = controller.select_best(
        features,
        domain_hint=args.domain_hint,
        mode="experimental_vote",
        return_details=True,
    )
    timings["controller_seconds"] = round(time.time() - t3, 3)

    t4 = time.time()
    optional_test_results = _run_optional_tests(candidates, args.optional_test_json)
    timings["optional_tests_seconds"] = round(time.time() - t4, 3)

    selected_index = int(selection_result.get("selected_index", -1)) if isinstance(selection_result, dict) else int(selection_result)
    selected_text = candidates[selected_index]["text"] if 0 <= selected_index < len(candidates) else ""
    exact_shapes = [list(row.shape) for row in feature_rows]
    verdict = "PASS"
    if len(candidates) < 2 or selected_index < 0:
        verdict = "FAIL"
    elif any(shape != [3, 4, 2048] for shape in exact_shapes):
        verdict = "FAIL"
    elif diagnostic_all is None or experimental_vote is None:
        verdict = "PARTIAL"

    timings["total_seconds"] = round(time.time() - t0, 3)
    return {
        "verdict": verdict,
        "prompt": prompt,
        "domain_hint": args.domain_hint,
        "generation_params": {
            "n_candidates": args.n_candidates,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "mode": args.mode,
        "candidates": candidates,
        "feature_shapes": exact_shapes,
        "selected_index": selected_index,
        "selected_text": selected_text,
        "selection_result": selection_result,
        "conservative_result": conservative_result,
        "diagnostic_all_scores": diagnostic_all,
        "experimental_vote_diagnostic": experimental_vote,
        "optional_test_results": optional_test_results,
        "warnings": warnings,
        "timings": timings,
        "device_info": _device_info(args.device),
    }


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    warnings: list[str] = []
    prompt = _load_prompt(args)
    try:
        try:
            payload = _run_once(args, prompt, warnings)
        except BaseException as exc:
            if not _is_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            original = {"n_candidates": args.n_candidates, "max_new_tokens": args.max_new_tokens}
            args.n_candidates = min(args.n_candidates, 2)
            args.max_new_tokens = min(args.max_new_tokens, 256 if original["max_new_tokens"] > 192 else 96)
            warnings.append(f"CUDA OOM retry: reduced generation from {original} to n={args.n_candidates}, max_new_tokens={args.max_new_tokens}")
            payload = _run_once(args, prompt, warnings)
    except Exception as exc:
        payload = {
            "verdict": "FAIL",
            "prompt": prompt,
            "domain_hint": args.domain_hint,
            "generation_params": {
                "n_candidates": args.n_candidates,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            },
            "mode": args.mode,
            "candidates": [],
            "feature_shapes": [],
            "selected_index": -1,
            "selected_text": "",
            "warnings": warnings,
            "device_info": _device_info(args.device),
            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }

    _write_outputs(payload, output)
    print(f"BG_TRANSFORMER_BEST_OF_N_SMOKE_VERDICT = {payload['verdict']}")
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(PROJECT_ROOT)}")
    print(f"Wrote {AGG_JSON.relative_to(PROJECT_ROOT)}")
    return 0 if payload["verdict"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
