"""CPU check of entropy_pass.score_batch's token alignment, with a toy causal model (no GPU, no Ouro weights).

The toy model's next-token logits depend on the current token only. Answers are sampled from it with the same
warpers as generation, storing log-probabilities as HF generate does; score_batch must then reproduce them
(lp_check_mad ~ 0) and the entropies must equal a direct computation. An off-by-one fails both.
"""
import types

import torch

import entropy_pass as E


class Toy(torch.nn.Module):
    def __init__(self, V=50):
        super().__init__()
        self.emb = torch.nn.Embedding(V, V)
        torch.nn.init.normal_(self.emb.weight, std=3.0)

    def forward(self, input_ids, use_cache=False, exit_at_step=None):
        return types.SimpleNamespace(logits=self.emb(input_ids))


torch.manual_seed(0)
model = Toy()
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper  # noqa: E402
warp = LogitsProcessorList([TemperatureLogitsWarper(E.TEMP), TopKLogitsWarper(20), TopPLogitsWarper(E.TOP_P)])
items, want = [], []
with torch.no_grad():
    for n_prompt, n_gen in ((5, 12), (9, 3), (3, 30), (7, 0)):
        p = torch.randint(0, 50, (n_prompt,)).tolist()
        g, lp, ents = [], [], []
        last = p[-1]
        for _ in range(n_gen):
            z = model.emb.weight[last][None]
            ents.append(float(-(torch.log_softmax(z, -1).exp() * torch.log_softmax(z, -1)).sum()))
            w = torch.log_softmax(warp(None, z.clone()), -1)
            tok = int(torch.multinomial(w.exp(), 1))
            g.append(tok); lp.append(float(w[0, tok])); last = tok
        items.append((p, g, lp, 8))
        want.append(ents)
    out = E.score_batch(model, warp, items, pad_id=0)
for (ef, ep, n_pre, mad), ents, (_, g, _, n_star) in zip(out, want, items):
    if not g:
        assert n_pre == 0
        continue
    assert mad < 1e-5, mad
    assert abs(ef - sum(ents) / len(ents)) < 1e-4 and abs(ep - sum(ents[:n_star]) / len(ents[:n_star])) < 1e-4
    assert n_pre == min(n_star, len(g))
print("entropy_pass alignment check passed on", len(items), "toy answers (padded batch, one empty answer)")
