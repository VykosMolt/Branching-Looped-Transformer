"""Post-hoc: unembed the loop-4 output at each prefix position (the model's own next-token distribution).
Does it favour the answer's eventual verdict over the other verdicts, at the cut or anywhere before it?"""
import sys, json, collections, torch
R1 = "/home/moloch/ouro_project/opi/preanswer/fixed_prefix_rerun_20260927"
ROOT = "/home/moloch/ouro_project"
sys.path[:0] = [R1, ROOT, ROOT + "/shared"]
import analyze_run1 as A
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor
recs = A.load(); A.answer_free_table(recs)
N = 48
primary = [r for r in recs if not r["malformed"] and r["free"][N]]
ext = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
tok = ext.tokenizer
def first_ids(words):
    return sorted({tok(w, add_special_tokens=False).input_ids[0] for w in words})
WORD = {v: first_ids([v, " " + v, v.lower(), " " + v.lower()]) for v in ("True", "False", "Unknown")}
LETTER = {l: first_ids([l, " " + l]) for l in ("A", "B", "C")}
def verdict_ids(r, v):  # word tokens for verdict v plus the letter of the option whose text is v
    ids = set(WORD[v])
    for l, text in r["options"].items():
        if text.strip().rstrip(".").capitalize() == v and l in LETTER:
            ids |= set(LETTER[l])
    return sorted(ids)
stats = collections.defaultdict(list)
for i, r in enumerate(primary):
    v_true = r["options"][r["parsed_answer"]].strip().rstrip(".").capitalize()
    others = [v for v in ("True", "False", "Unknown") if v != v_true]
    ids = r["prompt_ids"] + r["gen_ids"][:N]; plen = len(r["prompt_ids"])
    x = torch.tensor([ids], device=ext.device)
    with torch.inference_mode():
        p = torch.softmax(ext.model(input_ids=x, use_cache=False).logits[0, plen - 1:plen + N].float(), -1)  # rows: before gen token 0 .. after the cut
    pe = p[:, verdict_ids(r, v_true)].sum(-1)
    po = torch.stack([p[:, verdict_ids(r, v)].sum(-1) for v in others]).mean(0)
    top = p.argmax(-1).tolist()
    stats["p_eventual_at_cut"].append(float(pe[-1])); stats["p_other_at_cut"].append(float(po[-1]))
    stats["top_is_eventual_at_cut"].append(top[-1] in verdict_ids(r, v_true))
    stats["top_is_any_verdict_at_cut"].append(any(top[-1] in verdict_ids(r, v) for v in ("True", "False", "Unknown")))
    stats["max_p_eventual_in_prefix"].append(float(pe.max())); stats["max_p_other_in_prefix"].append(float(po.max()))
    stats["positions_top_eventual"].append(sum(t in verdict_ids(r, v_true) for t in top))
    stats["positions_top_other"].append(sum(any(t in verdict_ids(r, v) for v in others) for t in top) / 2)
    stats["success"].append(bool(r["success"]))
    if i % 200 == 0:
        print(i, len(primary), flush=True)
n = len(primary)
mean = lambda k: sum(stats[k]) / n
summary = {"n": n,
           "P(eventual verdict) at cut, mean": mean("p_eventual_at_cut"), "P(each other verdict) at cut, mean": mean("p_other_at_cut"),
           "top token at cut is the eventual verdict": mean("top_is_eventual_at_cut"),
           "top token at cut is any verdict": mean("top_is_any_verdict_at_cut"),
           "max over prefix P(eventual), mean": mean("max_p_eventual_in_prefix"), "max over prefix P(other), mean": mean("max_p_other_in_prefix"),
           "positions per answer whose top token is the eventual verdict": mean("positions_top_eventual"),
           "positions per answer whose top token is another verdict (per verdict)": mean("positions_top_other")}
from proto_introspection_controls_analysis import auroc
y = torch.tensor([1.0 if s else 0.0 for s in stats["success"]])
summary["AUROC for correctness of P(eventual) at cut"] = auroc(torch.tensor(stats["p_eventual_at_cut"]), y)
summary["AUROC for correctness of max-over-prefix P(eventual)"] = auroc(torch.tensor(stats["max_p_eventual_in_prefix"]), y)
print(json.dumps(summary, indent=2))
json.dump({"summary": summary, "per_answer": stats}, open(sys.argv[1], "w"))
