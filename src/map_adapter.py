"""
MAP (Magnitude-And-direction, global) adapter.

The paper's MAP method [2] takes a *global* view of a weight matrix: flatten
W in R^{d_out x d_in} to a single vector w in R^k (k = d_out*d_in) and write

    w' = alpha * w_hat + beta * dw_hat

where  w_hat  = vec(W0) / ||vec(W0)||      (frozen unit direction of pretrained W)
       dw_hat = vec(dW) / ||vec(dW)||       (learnable unit perturbation direction)
       alpha, beta  are learnable scalars.

This is the geometric opposite of DoRA's column-wise decomposition: MAP rotates
the *entire* flattened matrix as one vector (a single global angle Theta_global),
whereas DoRA lets each column rotate independently (per-column angles theta_j).

peft has no MAP implementation, so we provide a standalone nn.Module that swaps
in for the target nn.Linear layers. To match LoRA/DoRA in trainable-parameter
count, dW is parameterized low-rank: dW = B @ A with B in R^{d_out x r},
A in R^{r x d_in} (so MAP has r*(d_out+d_in) + 2 trainable params per layer vs
LoRA's r*(d_out+d_in)).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .train import TARGET_MODULES


class MAPLinear(nn.Module):
    """Drop-in replacement for nn.Linear implementing the global MAP update.

    The pretrained weight W0 and bias are frozen buffers. Trainable parameters
    are the low-rank factors (A, B) of the perturbation direction and the two
    global scalars (alpha, beta). At init alpha = ||W0||_F and beta = 0, so the
    effective weight equals W0 exactly (training starts from the pretrained model).
    """

    def __init__(self, linear: nn.Linear, rank: int = 8):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.d_out, self.d_in, self.r = d_out, d_in, rank

        # Frozen pretrained weight / bias.
        self.register_buffer("W0", linear.weight.detach().clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

        # Frozen unit direction of the pretrained weight (flattened).
        w0_norm = self.W0.norm()  # Frobenius == L2 of the flattened vector
        self.register_buffer("w0_hat", (self.W0 / (w0_norm + 1e-12)))

        # Low-rank factors of the perturbation direction dW = B @ A.
        # Both initialized nonzero so dW (and hence dw_hat) is well-defined;
        # beta = 0 keeps W' = W0 at start, and beta's gradient is nonzero on the
        # first step (it sees dw_hat), which then unlocks gradients to A, B.
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.empty(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.normal_(self.B, std=1.0 / rank)

        # Global learnable scalars. alpha preserves the pretrained scale.
        self.alpha = nn.Parameter(w0_norm.clone().float())
        self.beta = nn.Parameter(torch.zeros(()))

    def effective_weight(self) -> torch.Tensor:
        """Return W' = alpha * w0_hat + beta * dw_hat as a (d_out, d_in) tensor."""
        dW = self.B @ self.A                       # (d_out, d_in), the raw perturbation
        dw_hat = dW / (dW.norm() + 1e-12)          # unit global direction
        W = self.alpha * self.w0_hat + self.beta * dw_hat
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"d_out={self.d_out}, d_in={self.d_in}, rank={self.r} (MAP)"


def _is_target(name: str) -> bool:
    """Suffix match, identical to what peft targets for LoRA/DoRA.

    Matching on the trailing component(s) (e.g. '.query', '.output.dense') keeps
    the target set to the 72 attention/FFN projections and excludes lookalikes
    such as the randomly-initialized 'classifier.dense' head.
    """
    return any(name == t or name.endswith("." + t) for t in TARGET_MODULES)


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace the submodule at a dotted path, handling ModuleList integer indices."""
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def build_map_model(model_name: str, num_labels: int, rank: int = 8):
    """Load the base model, freeze it, and swap target Linears for MAPLinear.

    The classification head is left trainable (mirroring peft's SEQ_CLS, which
    trains modules_to_save), so the head can adapt alongside the MAP directions.
    Returns the model with only MAP factors + scalars + classifier trainable.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    for p in model.parameters():
        p.requires_grad_(False)

    # Collect target Linear modules first (can't mutate while iterating).
    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and _is_target(name)
    ]
    for name, lin in targets:
        _set_submodule(model, name, MAPLinear(lin, rank=rank))

    # Keep the classification head trainable.
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad_(True)

    return model


def map_layer_weights(model) -> dict[str, tuple]:
    """Return {layer_name: (W0, W_final)} numpy arrays for every MAPLinear."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, MAPLinear):
            W0 = mod.W0.detach().cpu().numpy().copy()
            Wf = mod.effective_weight().detach().cpu().numpy().copy()
            out[name] = (W0, Wf)
    return out
