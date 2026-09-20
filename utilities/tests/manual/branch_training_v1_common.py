"""Shared harness for branch_training_logic_expansion_terminal_v1 (Parts E–R).

Builds branch pools (cheap candidate pools + bounded, resumable model-generated pools),
labels them with EXTERNAL verifiers only, and provides resumable/pausable training helpers.
Reuses corecontent_v2 (io/dataset/feature) and the logic generators/verifiers.

Hard rules: correctness labels come only from external verifiers; DualAnchor/CoreContent are
policy/soft teachers, never correctness; no base-checkpoint/tokenizer/registry overwrite;
science diagnostic-only; train only under the model root; never train on heldout.
"""
from __future__ import annotations
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bg_corecontent_v2_common as v2  # noqa: E402
import branch_training_logic_v1_common as L  # noqa: E402

PROJECT_ROOT = v2.PROJECT_ROOT
OUT_ROOT = v2.PROBE_ROOT / "branch_training_logic_expansion_terminal_v1_2026-06-06"
DATA_ROOT = PROJECT_ROOT / "shared/data/branch_training_logic_expansion_v1"
PROC = DATA_ROOT / "processed"
TEACHER = DATA_ROOT / "teacher"
TRAIN = DATA_ROOT / "train"
MODEL_ROOT = PROJECT_ROOT / "opi/taps/models/branch_training_logic_expansion_v1"
PROGRESS = OUT_ROOT / "progress"
GEN_DIR = PROC / "gen_shards"

CORE_DOMAINS = v2.CORE_DOMAINS
write_md, write_csv, write_json = v2.write_md, v2.write_csv, v2.write_json
read_jsonl, write_jsonl = v2.read_jsonl, v2.write_jsonl
status_line, md_table, fmt, finite_mean = v2.status_line, v2.md_table, v2.fmt, v2.finite_mean


def ensure_dirs() -> None:
    for d in (OUT_ROOT, PROGRESS, PROC, TEACHER, TRAIN, MODEL_ROOT, GEN_DIR):
        d.mkdir(parents=True, exist_ok=True)


def prog(name: str, payload: dict) -> None:
    ensure_dirs()
    (PROGRESS / f"{name}.json").write_text(json.dumps({**payload, "saved_at": time.time()}, default=v2.json_default) + "\n")


# ===================================================== task sources
def load_logic_tasks() -> list[dict[str, Any]]:
    return read_jsonl(PROC / "logic_tasks.jsonl")


def load_core_tasks(max_per_domain: int = 4000) -> list[dict[str, Any]]:
    """Reuse corecontent_v2 deduped candidate groups (text + external labels) as branch tasks."""
    p = PROJECT_ROOT / "shared/data/corecontent_v2/processed/candidate_groups_deduped.jsonl"
    rows = read_jsonl(p)
    by_dom: dict[str, list] = defaultdict(list)
    for g in rows:
        if g["domain"] in ("coding", "math", "reasoning", "alignment"):
            by_dom[g["domain"]].append(g)
    out = []
    for dom, gs in by_dom.items():
        gs.sort(key=lambda g: L._sid("bt_core", g["group_uid"]))
        for g in gs[:max_per_domain]:
            out.append(_core_group_to_task(g))
    return out


def _core_group_to_task(g: dict[str, Any]) -> dict[str, Any]:
    dom = g["domain"]
    pos = next((c for c in g["candidates"] if c["reward"] > 0), None)
    label_type = {"coding": "unit_tests", "math": "exact_answer", "reasoning": "mcq_answer_key",
                  "alignment": "preference_label"}[dom]
    return {"task_uid": g["group_uid"], "dataset": g["source_dataset"], "category": dom, "domain": dom,
            "task_type": dom, "split": g["split"], "label_type": label_type,
            "candidates": g["candidates"], "kind": g.get("kind"),
            "gold_answer": (pos.get("candidate_kind") if pos else None)}


# ===================================================== cheap candidate pools (selection pools)
def cheap_branch_pool(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Branch attempts from dataset/solver sources (no model gen). External labels only."""
    cat = task["category"]; lt = task["label_type"]
    branches = []
    if "candidates" in task:  # reused core-domain group: candidates already carry rewards
        for i, c in enumerate(task["candidates"]):
            r = float(c["reward"])
            branches.append(_battempt(f"b{i}", c.get("candidate_text", c.get("candidate_uid", "")),
                                      "dataset_option" if c.get("candidate_kind") in ("chosen", "rejected", "mcq_option") else "mutated",
                                      "pass" if r > 0 else "fail", r, _fmt_final(c), True,
                                      bfmt="direct_answer"))
        return branches
    # logic task: options
    opts = task.get("options") or []
    ak = task.get("answer_key")
    for j, o in enumerate(opts):
        letter = chr(65 + j)
        txt = f"Answer: {letter}. {o}"
        passed = (j == ak)
        branches.append(_battempt(f"b{j}", txt, "dataset_option", "pass" if passed else "fail",
                                  1.0 if passed else 0.0, str(o), True, bfmt="direct_answer"))
    return branches


def _battempt(bid, text, source, label, reward, final, parse_ok, bfmt="direct_answer", strategy="given",
              lineage=None) -> dict[str, Any]:
    return {"branch_id": bid, "branch_text": text, "branch_format": bfmt, "strategy_label": strategy,
            "source": source, "external_label": label, "objective_reward": float(reward),
            "verifier_result": {}, "final_answer": final, "parse_ok": bool(parse_ok),
            "failure_modes": [], "lineage": lineage or {"generation_method": "dataset", "sample_seed": 0}}


def _fmt_final(c: dict[str, Any]) -> str:
    return str(c.get("candidate_kind", ""))[:32]


def make_group(task: dict[str, Any], branches: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [b["objective_reward"] for b in branches]
    has_pos = any(r > 0 for r in rewards)
    mx = max(rewards) if rewards else 0.0
    return {"group_id": f"branch_v1_{task['split']}_{task['domain']}_{task['dataset']}_{L._sid(task['task_uid'])%10**8}",
            "split": task["split"], "domain": task["domain"], "dataset": task["dataset"],
            "task_id": task["task_uid"], "task_type": task.get("task_type"), "task_prompt": task.get("task_prompt", ""),
            "label_type": task["label_type"], "verifier": task.get("verifier", {"type": task["label_type"]}),
            "branch_attempts": branches,
            "external_oracle": {"has_positive_oracle": has_pos,
                                "best_branch_ids": [b["branch_id"] for b in branches if b["objective_reward"] >= mx and mx > 0],
                                "reward_diverse": len(set(r > 0 for r in rewards)) > 1,
                                "all_wrong": not has_pos, "all_correct": all(r > 0 for r in rewards) and bool(rewards)},
            "teacher_traces": {"dualanchor": None, "corecontent_v2": None},
            "quality_flags": {}, "provenance": {"category": task["category"]}}


# ===================================================== external verifiers for generated branches
def verify_final(task: dict[str, Any], final_answer: str) -> tuple[str, float]:
    """Return (label, reward) for a parsed final answer using the task's external verifier."""
    if final_answer is None:
        return "unknown", 0.0
    fa = str(final_answer).strip()
    lt = task["label_type"]; cat = task["category"]
    if lt in ("mcq_answer_key", "deterministic_rubric", "constraint_solver") or task.get("options"):
        opts = task.get("options") or []
        ak = task.get("answer_key")
        gold = task.get("gold_answer")
        # accept letter, index, or option-text / gold-token match
        m = re.search(r"\b([A-Ga-g])\b", fa)
        if m and ak is not None:
            idx = ord(m.group(1).upper()) - 65
            return ("pass", 1.0) if idx == ak else ("fail", 0.0)
        low = fa.lower()
        for tok in ("true", "false", "unknown", "valid", "invalid"):
            if tok in low and gold:
                return ("pass", 1.0) if tok == str(gold).lower() else ("fail", 0.0)
        if gold and str(gold).lower() in low:
            return "pass", 1.0
        return "fail", 0.0
    if lt == "exact_answer":
        gold = task.get("gold_answer", "")
        nums = re.findall(r"-?\d[\d,]*\.?\d*", fa)
        cand = nums[-1].replace(",", "") if nums else fa
        ok = math_equal_robust(cand, gold) or math_equal_robust(fa, gold)  # LaTeX/symbolic-aware
        return ("pass", 1.0) if ok else ("fail", 0.0)
    return "unknown", 0.0


_FINAL_RE = re.compile(r"final answer[:\s]*([^\n]+)", re.IGNORECASE)


def _latex_norm(s: object) -> str:
    t = str(s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\ ", ""), ("$", ""),
                 ("\\pi", "pi"), ("π", "pi"), ("\\cdot", "*"), ("\\times", "*"), ("\\div", "/"),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("{", "{"), ("\\%", ""), ("%", "")):
        t = t.replace(a, b)
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\frac(\d)(\d)", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", t)
    t = t.replace("^", "**").replace("{", "(").replace("}", ")")
    return t.strip().strip("=").strip()


def math_equal_robust(pred: object, gold: object) -> bool:
    """Numeric/string match (v2) + LaTeX-normalized sympy symbolic equivalence (fixes \\frac{41\\pi}{4} == 41pi/4)."""
    if pred is None:
        return False
    if v2.math_equal(pred, gold):
        return True
    pn, gn = _latex_norm(pred), _latex_norm(gold)
    if pn.replace(" ", "") == gn.replace(" ", "") and pn:
        return True
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        tr = standard_transformations + (implicit_multiplication_application,)
        p = parse_expr(pn, transformations=tr); g = parse_expr(gn, transformations=tr)
        return bool(sympy.simplify(p - g) == 0)
    except Exception:
        return False


_TRIM_MARKERS = ("\nProblem", "\nQuestion:", "\nExercise", "\nassistant", "\nuser",
                 "<|im_end|>", "<|endoftext|>", "\n\n\n\n")


def trim_generation(text: str) -> str:
    """Fast post-pass (replaces slow per-token stop_strings): cut drift/new-problems/fake turns/
    newline padding, and end just after the first complete 'FINAL ANSWER: ...' line."""
    cut = len(text)
    for mark in _TRIM_MARKERS:
        i = text.find(mark)
        if i != -1:
            cut = min(cut, i)
    text = text[:cut]
    m = _FINAL_RE.search(text)
    if m:
        text = text[:m.end()]  # stop right after the first committed final answer
    return text.strip()


def parse_final(text: str) -> str | None:
    m = _FINAL_RE.search(text)  # first commitment (before any degenerate repetition)
    if m:
        a = m.group(1).strip()
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        a = lines[-1] if lines else None
    if a is None:
        return None
    a = a.replace("<answer>", "").replace("</answer>", "")
    a = a.strip().strip("*`").strip().strip("<>").strip()  # strip wrappers like <200000>, <A>, **..**
    return a[:80] or None


# ===================================================== branch scaffolds (structural diversity)
SCAFFOLDS = {
    "direct": "Solve the problem directly and concisely. End with 'Final answer: <answer>'.",
    "case_split": "Solve by splitting into cases and checking each. End with 'Final answer: <answer>'.",
    "counterexample": "Try to find a counterexample; if none exists, conclude it holds. End with 'Final answer: <answer>'.",
    "elimination": "Eliminate the options that cannot hold, one by one. End with 'Final answer: <answer>'.",
    "formalize": "First restate the facts formally, then reason. End with 'Final answer: <answer>'.",
    "self_check": "Solve, then double-check your reasoning for errors. End with 'Final answer: <answer>'.",
}
SCAFFOLD_FMT = {"direct": "direct_answer", "case_split": "case_split", "counterexample": "counterexample_search",
                "elimination": "option_elimination", "formalize": "constraint_table", "self_check": "self_check"}


# tool-free answer-forcing system prompt, adapted from the local-agent OURO_SOLVER_POSTURE.
# NO tools, NO external experts/oracles, NO code execution — pure model reasoning, forced to commit.
GEN_SYSTEM = (
    "You are Ouro, a compact looped reasoning model. Solve from your own competence. You have NO tools, "
    "no internet, and no external help — reason it out yourself.\n"
    "- Bind the object that determines the answer (the equation, rule, invariant, or decisive case).\n"
    "- Name the tempting wrong route and why it fails; treat a wrong answer as a route error, not a local bug.\n"
    "- Be concise. Do NOT output thinking tags. Do NOT invent new problems, questions, or exercises.\n"
    "- Answer ONLY the problem given, then stop.\n"
    "- End your response with exactly one final line: 'FINAL ANSWER: <answer>'."
)
GEN_SYSTEM_MATH = GEN_SYSTEM + (
    "\nFor this math problem: show only the necessary steps, then commit to a single exact answer. "
    "The last line MUST be 'FINAL ANSWER: <number>' and nothing after it.")
GEN_SYSTEM_CODE = (
    "You are Ouro, a compact looped reasoning model with NO tools. Write a correct Python solution from your own "
    "competence. Identify the invariant and exact return value, then give the COMPLETE function in one ```python "
    "code block. No TODOs, no prose inside the code. Do not invent extra problems.")


_FNAME_RE = re.compile(r"(?:assert|assert_equal|assertEqual)\s*\(?\s*([A-Za-z_]\w*)\s*\(")


def _required_fn_name(tests) -> str | None:
    for t in (tests or []):
        m = _FNAME_RE.search(str(t))
        if m:
            return m.group(1)
    return None


def gen_messages(task: dict[str, Any], scaffold: str) -> tuple[str, str]:
    """Return (system, user) for the tool-free, answer-forcing generation prompt."""
    body = task.get("task_prompt", "")
    lt = task.get("label_type")
    if lt == "unit_tests":
        fn = _required_fn_name(task.get("unit_tests"))
        if fn:
            body += f"\n\nYour function MUST be named exactly `{fn}` (the tests call `{fn}`)."
        return GEN_SYSTEM_CODE, body
    if task.get("options"):
        body += "\nOptions:\n" + "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(task["options"]))
    sysp = GEN_SYSTEM_MATH if lt == "exact_answer" else GEN_SYSTEM
    return sysp, f"{body}\n\n{SCAFFOLDS[scaffold]}"


def build_prompt(tok, task: dict[str, Any], scaffold: str) -> str:
    sysp, user = gen_messages(task, scaffold)
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"{sysp}\n\n{user}\n\nFINAL ANSWER:" if False else f"{sysp}\n\n{user}\n"


# ===================================================== resumable / pausable generation
def stop_requested(job: str) -> bool:
    return (MODEL_ROOT / job / "STOP").exists() or (OUT_ROOT / "STOP").exists()


# domain-aware generation budget: math needs lots of room (the model yaps); short-answer domains do not.
DOMAIN_MAXTOK = {"math": 1400, "coding": 640, "logic": 384, "reasoning": 320}
# stop the model from drifting into inventing NEW problems (the main math/logic yap failure mode)
STOP_STRINGS = ["\nProblem", "\n\nProblem", "\nQuestion:", "\n\nQuestion:", "\nExercise", "\n\nExercise",
                "\nProblem:", "\n\nQ:", "\nQ:",
                # halt degenerate post-answer repetition / fake new chat turns
                "\nassistant", "\nuser", "\nassistant\n", "```\nassistant", "<|im_end|>", "<|endoftext|>",
                "\n\n\n\n"]  # cut endless-newline padding after the answer (saves math budget)
BRANCH_TEXT_CAP = 4000  # store enough of long math derivations for training (verification uses full text)


_FA_LINE = re.compile(r"final answer\s*:[^\n]+\n", re.I)
_DRIFT = ("\nProblem", "\nQuestion:", "\nExercise", "\nassistant", "\nuser", "<|im_end|>", "<|endoftext|>")


def _make_stop(tok, plen: int):
    """Cheap early-stop: decode only the last ~64 tokens every 16 steps; stop a sequence once it has
    written a complete 'FINAL ANSWER: ...' line or drifted into a new problem/turn. Per-row (batch-safe)."""
    import torch
    from transformers import StoppingCriteria

    class _S(StoppingCriteria):
        def __call__(self, input_ids, scores=None, **kw):
            res = []
            for row in input_ids:
                n = int(row.shape[0])
                if (n - plen) < 8 or (n - plen) % 16 != 0:
                    res.append(False); continue
                tail = tok.decode(row[max(plen, n - 64):], skip_special_tokens=True)
                res.append(bool(_FA_LINE.search(tail)) or any(d in tail for d in _DRIFT))
            return torch.tensor(res, device=input_ids.device)
    return _S()


def _gen_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mp = PROJECT_ROOT / "shared/models/ouro_rltt_local"
    tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(mp), torch_dtype="auto", trust_remote_code=True,
                                                 local_files_only=True, low_cpu_mem_usage=True).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _gen_manifest_path():
    return GEN_DIR / "gen_manifest.json"


def _load_gen_manifest() -> dict:
    p = _gen_manifest_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"shards": [], "completed_task_uids": []}


def generate_branch_pools(tasks: list[dict[str, Any]], k: int = 4, max_new_tokens: int = 160,
                          job: str = "branch_gen", shard_groups: int = 40, time_budget_s: float = 1e18) -> dict:
    """Resumable + pausable model branch generation. STOP sentinel or time budget -> graceful save+exit."""
    import torch
    ensure_dirs()
    man = _load_gen_manifest()
    done = set(man.get("completed_task_uids", []))
    pending = [t for t in tasks if t["task_uid"] not in done]
    print(f"  generate: {len(tasks)} tasks, {len(pending)} pending, {len(done)} done", flush=True)
    if not pending:
        return man
    model, tok = _gen_model()
    scaffolds = ["direct", "case_split", "counterexample", "elimination", "formalize", "self_check"]
    buf = []
    started = time.time(); t_log = started; enc_groups = 0; shard_idx = max([s["idx"] for s in man["shards"]], default=-1)

    def flush():
        nonlocal shard_idx
        if not buf:
            return
        shard_idx += 1
        fp = GEN_DIR / f"gen_{shard_idx:04d}.jsonl"
        write_jsonl(fp, buf)
        man["shards"].append({"file": fp.name, "idx": shard_idx, "groups": len(buf)})
        for g in buf:
            done.add(g["task_id"])
        man["completed_task_uids"] = sorted(done)
        _gen_manifest_path().write_text(json.dumps(man, default=v2.json_default) + "\n")
        prog("E_generate", {"shards": len(man["shards"]), "groups_done": len(done), "pending": len(pending)})
        print(f"    wrote {fp.name}: {len(buf)} groups (total {len(done)})", flush=True)
        buf.clear()

    paused = False
    for ti, task in enumerate(pending):
        if stop_requested(job) or (time.time() - started) > time_budget_s:
            paused = True
            print(f"  pause requested (STOP/time); flushing at {len(done)} done", flush=True)
            break
        branches = []
        # rotate scaffold pairs across tasks for corpus-level diversity; 2 scaffolds x ceil(k/2) seqs
        rot = L._sid("scaf", task["task_uid"]) % len(scaffolds)
        n_scaf = 2 if k <= 4 else 3
        use = [scaffolds[(rot + j) % len(scaffolds)] for j in range(n_scaf)]
        per = max(1, -(-k // len(use)))  # ceil
        bi = 0
        for sc in use:
            prompt = build_prompt(tok, task, sc)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
            mnt = DOMAIN_MAXTOK.get(task["domain"], max_new_tokens)
            plen = enc["input_ids"].shape[1]
            # one sequence per call: Ouro's 4-loop KV cache is ~4x normal; batching seqs at 1400 math
            # tokens OOMs a 12GB card. Single-seq + cache clear keeps peak bounded.
            from transformers import StoppingCriteriaList
            stopper = StoppingCriteriaList([_make_stop(tok, plen)])
            for rep in range(per):
                try:
                    with torch.inference_mode():
                        out = model.generate(**enc, max_new_tokens=mnt, do_sample=True,
                                             temperature=0.8, top_p=0.95, num_return_sequences=1,
                                             pad_token_id=tok.pad_token_id, stopping_criteria=stopper)
                    text = trim_generation(tok.decode(out[0][plen:], skip_special_tokens=True))
                    del out
                except Exception:
                    text = None
                if not text:
                    continue
                label, reward, fa, parse_ok = label_branch(task, text)
                branches.append(_battempt(f"b{bi}", text[:BRANCH_TEXT_CAP], "model_generated", label, reward, fa,
                                          parse_ok, bfmt=SCAFFOLD_FMT[sc], strategy=sc,
                                          lineage={"generation_method": f"sample_{sc}", "sample_seed": rep,
                                                   "temperature": 0.8, "top_p": 0.95, "prompt_variant": sc}))
                bi += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(branches) < 2:
            done.add(task["task_uid"]); continue
        buf.append(make_group(task, branches))
        enc_groups += 1
        if len(buf) >= shard_groups:
            flush()
        if enc_groups % 5 == 0 or (time.time() - t_log) > 90:
            t_log = time.time()
            rate = enc_groups / (time.time() - started + 1e-9)
            print(f"  ... gen groups={enc_groups} ({rate:.3f} grp/s) done={len(done)} buf={len(buf)}", flush=True)
    flush()
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass
    man["paused"] = paused
    _gen_manifest_path().write_text(json.dumps(man, default=v2.json_default) + "\n")
    return man


def load_generated_groups() -> list[dict[str, Any]]:
    man = _load_gen_manifest()
    out = []
    for s in man.get("shards", []):
        out += read_jsonl(GEN_DIR / s["file"])
    return out


# ===================================================== multi-domain generation tasks
def _reconstruct_core_gen(g: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild a generation-ready, externally-verifiable task from a v2 candidate group."""
    dom = g["domain"]
    cands = g["candidates"]
    pos = next((c for c in cands if c["reward"] > 0), None)
    if pos is None:
        return None
    if dom == "math":
        txt = pos.get("candidate_text", "")
        prompt = txt.split("\nAnswer:")[0].strip()
        gold = txt.split("\nAnswer:")[-1].strip()
        if not prompt or not gold:
            return None
        # hendrycks_math reuses row indices across its 7 subjects -> group_uid collides; hash the
        # actual prompt into the uid so distinct problems get distinct uids/group_ids.
        uid = f"{g['group_uid']}::{v2.text_hash(prompt)[:8]}"
        return {"task_uid": uid, "dataset": g["source_dataset"], "category": "math", "domain": "math",
                "task_type": "math", "split": g["split"], "label_type": "exact_answer",
                "task_prompt": prompt, "options": None, "gold_answer": gold, "verifier": {"type": "exact_answer"}}
    if dom == "reasoning":
        prompt = pos.get("candidate_text", "").split("\nAnswer:")[0].strip()
        opts, ak = [], None
        for i, c in enumerate(cands):
            seg = c.get("candidate_text", "").split("\nAnswer:")[-1].strip()
            opt = seg.split(". ", 1)[-1] if ". " in seg else seg
            opts.append(opt)
            if c["reward"] > 0:
                ak = i
        if ak is None or len(opts) < 2 or not prompt:
            return None
        uid = f"{g['group_uid']}::{v2.text_hash(prompt)[:8]}"  # unique-by-prompt (defensive; same fix as math)
        return {"task_uid": uid, "dataset": g["source_dataset"], "category": "reasoning", "domain": "reasoning",
                "task_type": "reasoning", "split": g["split"], "label_type": "mcq_answer_key",
                "task_prompt": prompt, "options": opts, "answer_key": ak, "gold_answer": opts[ak],
                "verifier": {"type": "mcq_answer_key"}}
    return None


def load_coding_gen_tasks(n: int = 600) -> list[dict[str, Any]]:
    """MBPP train tasks with unit tests (externally verifiable generated code)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    except Exception:
        return []
    out = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        tests = ex.get("test_list") or []
        if not tests:
            continue
        out.append({"task_uid": f"mbpp::train::{ex.get('task_id', i)}", "dataset": "google-research-datasets/mbpp",
                    "category": "coding", "domain": "coding", "task_type": "coding", "split": "train",
                    "label_type": "unit_tests", "task_prompt": ex.get("text", ""),
                    "unit_tests": tests, "test_setup": ex.get("test_setup_code", ""),
                    "gold_answer": ex.get("code", ""), "verifier": {"type": "unit_tests"}})
    return out


def load_gen_tasks_balanced(n_total: int = 1600, core_pool: int = 6000) -> list[dict[str, Any]]:
    """Domain-balanced generation set across logic/math/reasoning/coding (verifiable; alignment excluded)."""
    per = max(1, n_total // 4)
    # logic
    logic = [t for t in load_logic_tasks() if t["split"] == "train"]
    logic.sort(key=lambda t: L._sid("gen", t["category"], t["task_uid"]))
    # round-robin logic categories for structural diversity
    bycat: dict[str, list] = {}
    for t in logic:
        bycat.setdefault(t["category"], []).append(t)
    logic_sel = []
    while len(logic_sel) < per and any(bycat.values()):
        for c in list(bycat):
            if bycat[c]:
                logic_sel.append(bycat[c].pop())
                if len(logic_sel) >= per:
                    break
    # math + reasoning from v2 groups
    rows = read_jsonl(PROJECT_ROOT / "shared/data/corecontent_v2/processed/candidate_groups_deduped.jsonl")
    math_t, reas_t = [], []
    for g in rows:
        if g["domain"] == "math" and g["split"] == "train":
            r = _reconstruct_core_gen(g)
            if r:
                math_t.append(r)
        elif g["domain"] == "reasoning" and g["split"] == "train":
            r = _reconstruct_core_gen(g)
            if r:
                reas_t.append(r)
    math_t.sort(key=lambda t: L._sid("gen", t["task_uid"])); reas_t.sort(key=lambda t: L._sid("gen", t["task_uid"]))
    coding_t = load_coding_gen_tasks(min(per, 600))
    out = logic_sel[:per] + math_t[:per] + reas_t[:per] + coding_t[:per]
    return out


# ===================================================== code unit-test verifier (guarded)
_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _CODE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    # fallback: from first def/import to end
    idx = min([i for i in (text.find("def "), text.find("import ")) if i >= 0], default=-1)
    return text[idx:].strip() if idx >= 0 else text.strip()


def verify_code_unit_tests(code: str, tests: list[str], setup: str = "", timeout: float = 6.0) -> bool:
    import subprocess, tempfile
    if not code or "def " not in code:
        return False
    prog = (setup + "\n" if setup else "") + code + "\n" + "\n".join(tests) + "\n"
    try:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "cand.py"
            f.write_text(prog)
            r = subprocess.run(["python", str(f)], capture_output=True, timeout=timeout, cwd=td)
            return r.returncode == 0
    except Exception:
        return False


def label_branch(task: dict[str, Any], full_text: str) -> tuple[str, float, str, bool]:
    """Domain-aware external labelling of a generated branch -> (label, reward, final, parse_ok)."""
    if task["label_type"] == "unit_tests":
        code = extract_code(full_text)
        ok = verify_code_unit_tests(code, task.get("unit_tests", []), task.get("test_setup", ""))
        return ("pass" if ok else "fail", 1.0 if ok else 0.0, code[:80], bool(code and "def " in code))
    fa = parse_final(full_text)
    label, reward = verify_final(task, fa)
    return label, reward, (fa or ""), fa is not None
