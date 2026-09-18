# JumpReLU sparse autoencoder (Gemma Scope format), model-agnostic.
#
# A pretrained SAE decomposes one dense residual-stream activation into thousands
# of mostly-zero, hopefully-interpretable "features". Gemma Scope ships JumpReLU
# SAEs whose params.npz holds five arrays:
#   W_enc [d_model, d_sae]  W_dec [d_sae, d_model]
#   b_enc [d_sae]           b_dec [d_model]        threshold [d_sae]
#
# encode: pre = x @ W_enc + b_enc ; acts = (pre > threshold) * relu(pre)
# decode: acts @ W_dec + b_dec
# Each feature f owns a direction in residual space — its decoder row W_dec[f] —
# which is what makes an SAE feature a candidate steering vector.
#
# We load pretrained weights only (no training). `python -m general.sae` runs
# the self-tests (no model, no download).

from __future__ import annotations

from pathlib import Path

import torch


class JumpReLUSAE:
    """A pretrained JumpReLU SAE (plain tensors, not an nn.Module)."""

    def __init__(self, W_enc: torch.Tensor, W_dec: torch.Tensor, b_enc: torch.Tensor,
                 b_dec: torch.Tensor, threshold: torch.Tensor):
        self.W_enc, self.W_dec = W_enc, W_dec
        self.b_enc, self.b_dec, self.threshold = b_enc, b_dec, threshold
        self.d_model, self.d_sae = W_enc.shape

    def to(self, device, dtype: torch.dtype = torch.float32) -> "JumpReLUSAE":
        """Move all parameters to a device/dtype (returns self)."""
        self.W_enc = self.W_enc.to(device, dtype)
        self.W_dec = self.W_dec.to(device, dtype)
        self.b_enc = self.b_enc.to(device, dtype)
        self.b_dec = self.b_dec.to(device, dtype)
        self.threshold = self.threshold.to(device, dtype)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Residual activations -> sparse feature activations. x [..., d_model] -> [..., d_sae]."""
        x = x.to(self.W_enc.dtype)
        pre = x @ self.W_enc + self.b_enc
        return (pre > self.threshold) * torch.relu(pre)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Feature activations -> reconstructed residual. [..., d_sae] -> [..., d_model]."""
        return acts @ self.W_dec + self.b_dec

    def decoder_direction(self, feature: int) -> torch.Tensor:
        """Unit residual-space direction owned by `feature` (its decoder row)."""
        v = self.W_dec[feature]
        return v / v.norm()


def load_sae_from_npz(path: str | Path) -> JumpReLUSAE:
    """Load a Gemma Scope params.npz into a JumpReLUSAE (float32, on CPU)."""
    import numpy as np

    p = np.load(Path(path))
    t = lambda k: torch.from_numpy(p[k]).float()  # noqa: E731
    return JumpReLUSAE(t("W_enc"), t("W_dec"), t("b_enc"), t("b_dec"), t("threshold"))


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, d_sae = 16, 64
    # a toy SAE: encoder = random, decoder = encoder^T (typical tied-ish init), high threshold -> sparse
    W_enc = torch.randn(d_model, d_sae)
    sae = JumpReLUSAE(W_enc, W_enc.t().clone(), torch.zeros(d_sae),
                      torch.zeros(d_model), torch.full((d_sae,), 1.5))
    x = torch.randn(5, d_model)
    acts = sae.encode(x)
    assert acts.shape == (5, d_sae)
    assert (acts >= 0).all()                       # JumpReLU is non-negative
    assert (acts == 0).float().mean() > 0.5        # thresholding makes it sparse
    assert sae.decode(acts).shape == (5, d_model)  # decode returns to residual space
    assert abs(sae.decoder_direction(3).norm().item() - 1.0) < 1e-5
    # a feature below threshold everywhere stays exactly zero
    quiet = JumpReLUSAE(W_enc, W_enc.t().clone(), torch.zeros(d_sae),
                        torch.zeros(d_model), torch.full((d_sae,), 1e9))
    assert quiet.encode(x).abs().max() == 0
    print("general.sae self-tests passed")
