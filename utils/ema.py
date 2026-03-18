# SwiftDet - Model Exponential Moving Average
# MIT License - Original implementation
# Reference: Polyak & Juditsky 1992, "Acceleration of Stochastic Approximation
#            by Averaging"; Izmailov et al. 2018, "Averaging Weights Leads to
#            Wider Optima and Better Generalization" (SWA).
#
# EMA maintains a shadow copy of model parameters that is a running weighted
# average of the training-time parameters.  This yields smoother, more
# generalizable weights for evaluation.

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains exponential moving averages of model parameters.

    The EMA decay is ramped up from 0 to the target value during a warmup
    period so that early, rapidly changing parameters do not overly influence
    the shadow weights.

    Args:
        model: The model whose parameters will be tracked.
        decay: Target EMA decay rate (default 0.9999).
        warmup_steps: Number of update steps over which the decay ramps from
            0 to ``decay`` (default 2000).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 2000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0

        # Deep-copy model to create shadow parameters
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

    def _current_decay(self) -> float:
        """Compute the current decay rate with warmup ramp-up.

        Uses a cosine schedule to smoothly ramp from 0 to the target decay.
        """
        if self.step >= self.warmup_steps:
            return self.decay

        # Cosine ramp: 0 -> decay over warmup_steps
        progress = self.step / self.warmup_steps
        return self.decay * (1.0 - math.cos(math.pi * progress)) / 2.0

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters with the current model parameters.

        Args:
            model: The training model whose parameters are used for the update.
        """
        d = self._current_decay()
        self.step += 1

        model_params = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())

        for name, ema_param in self.ema_model.named_parameters():
            if name in model_params:
                # EMA update: ema = d * ema + (1 - d) * current
                ema_param.data.mul_(d).add_(model_params[name].data, alpha=1.0 - d)

        for name, ema_buf in self.ema_model.named_buffers():
            if name in model_buffers:
                # Copy buffers directly (e.g., batch-norm running stats)
                ema_buf.data.copy_(model_buffers[name].data)

    def apply(self, model: nn.Module):
        """Copy EMA parameters into the given model (typically for inference).

        Args:
            model: The model to receive the EMA parameters.
        """
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            model_param.data.copy_(ema_param.data)

        for ema_buf, model_buf in zip(self.ema_model.buffers(), model.buffers()):
            model_buf.data.copy_(ema_buf.data)

    def state_dict(self) -> dict:
        """Return the EMA state for checkpointing."""
        return {
            "ema_model": self.ema_model.state_dict(),
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "step": self.step,
        }

    def load_state_dict(self, state: dict):
        """Load EMA state from a checkpoint.

        Args:
            state: Dictionary previously returned by ``state_dict()``.
        """
        self.ema_model.load_state_dict(state["ema_model"])
        self.decay = state["decay"]
        self.warmup_steps = state["warmup_steps"]
        self.step = state["step"]
