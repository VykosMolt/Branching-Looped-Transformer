import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Low-rank learned attention pooling over token dimension."""

    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.query = nn.Parameter(torch.randn(attn_dim) * 0.01)

    def forward(self, hidden, attention_mask):
        keys = self.proj(hidden)
        scores = keys @ self.query
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("bs,bsd->bd", weights, hidden)


class PairwiseEvaluator(nn.Module):
    """
    Pairwise Constitutional Evaluator — the architecture that reached 70%.

    Reverted from v2 (pre-diff-norm regressed to 68%).
    Original design: normalize the DIFFERENCE, not the individual responses.
    """

    def __init__(self, hidden_dim=2048, gru_hidden=512, gru_layers=2,
                 intermediate_size=256, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden

        self.attention_pool = AttentionPool(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim, bias=False)
        self.input_proj = nn.Linear(hidden_dim, gru_hidden)

        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=False,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False
        )

        scorer_input_dim = gru_hidden * 2

        self.scorer = nn.Sequential(
            nn.LayerNorm(scorer_input_dim),
            nn.Linear(scorer_input_dim, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 1)
        )

    def forward(self, chosen_states, chosen_mask, rejected_states, rejected_mask):
        chosen_pooled = [self.attention_pool(h, chosen_mask) for h in chosen_states]
        rejected_pooled = [self.attention_pool(h, rejected_mask) for h in rejected_states]

        diffs = [c - r for c, r in zip(chosen_pooled, rejected_pooled)]

        normed = [self.input_norm(d) for d in diffs]
        projected = [self.input_proj(n) for n in normed]

        seq = torch.stack(projected, dim=0)
        _, final_hidden = self.gru(seq)
        gru_out = final_hidden[-1]

        combined = torch.cat([gru_out, projected[-1]], dim=-1)
        return self.scorer(combined)


def validate_hook_output(output):
    assert isinstance(output, (list, tuple)), \
        f"Expected list/tuple of hidden states, got {type(output)}"
    assert len(output) > 0, "Hook captured empty hidden states list"
    first = output[0]
    assert isinstance(first, torch.Tensor), \
        f"Expected tensor in hidden states list, got {type(first)}"
    assert first.dim() == 3, \
        f"Expected 3D tensor [batch, seq_len, hidden_dim], got {first.dim()}D"
    hidden_dim = first.shape[-1]
    for i, h in enumerate(output):
        assert h.shape[-1] == hidden_dim, \
            f"Hidden dim mismatch at step {i}: {h.shape[-1]} vs {hidden_dim}"


def test_pairwise():
    evaluator = PairwiseEvaluator()
    evaluator.eval()

    batch, seq_len, hidden_dim = 2, 128, 2048
    mask_c = torch.ones(batch, seq_len)
    mask_r = torch.ones(batch, seq_len)

    chosen = [torch.randn(batch, seq_len, hidden_dim) for _ in range(4)]
    rejected = [torch.randn(batch, seq_len, hidden_dim) for _ in range(4)]

    with torch.no_grad():
        score = evaluator(chosen, mask_c, rejected, mask_r)
    print(f"Forward pass — Output: {score.shape}")
    assert score.shape == (batch, 1)
    print("Forward pass: OK")

    with torch.no_grad():
        score_flipped = evaluator(rejected, mask_r, chosen, mask_c)
    print(f"Original: {score.squeeze().tolist()}")
    print(f"Flipped:  {score_flipped.squeeze().tolist()}")

    total_params = sum(p.numel() for p in evaluator.parameters())
    print(f"Total params: {total_params:,}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    test_pairwise()
