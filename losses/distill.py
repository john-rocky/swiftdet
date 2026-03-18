# SwiftDet - Knowledge distillation losses
# MIT License - Original implementation
#
# References:
#   - Hinton et al. 2015, "Distilling the Knowledge in a Neural Network"
#   - Feature distillation: Romero et al. 2015, "FitNets: Hints for Thin Deep Nets"

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillLoss(nn.Module):
    """Feature-based knowledge distillation loss.

    Aligns student intermediate features (typically from the neck/FPN) to the
    corresponding teacher features using mean squared error. When the student
    and teacher have different channel dimensions at a given level, a learnable
    1x1 adaptation convolution projects the student features to match the
    teacher channel count before computing MSE.

    Reference: Romero et al. 2015, "FitNets: Hints for Thin Deep Nets"

    Args:
        student_channels: List of channel sizes for each student feature level.
        teacher_channels: List of channel sizes for each teacher feature level.
        normalize: If True, L2-normalize features before computing MSE (default True).
    """

    def __init__(self, student_channels, teacher_channels, normalize=True):
        super().__init__()
        assert len(student_channels) == len(teacher_channels), (
            "Student and teacher must have the same number of feature levels."
        )
        self.normalize = normalize
        self.n_levels = len(student_channels)

        # Build 1x1 adaptation layers where channels differ
        self.adapts = nn.ModuleList()
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            if s_ch != t_ch:
                self.adapts.append(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
                )
            else:
                self.adapts.append(nn.Identity())

    def forward(self, student_features, teacher_features):
        """Compute feature distillation loss.

        Args:
            student_features: List of feature tensors from student neck.
                Each has shape (B, C_s_i, H_i, W_i).
            teacher_features: List of feature tensors from teacher neck.
                Each has shape (B, C_t_i, H_i, W_i).

        Returns:
            Scalar MSE distillation loss averaged over all levels.
        """
        total_loss = 0.0

        for i in range(self.n_levels):
            s_feat = self.adapts[i](student_features[i])
            t_feat = teacher_features[i].detach()

            if self.normalize:
                # L2-normalize along channel dimension for stable alignment
                s_feat = F.normalize(s_feat, p=2, dim=1)
                t_feat = F.normalize(t_feat, p=2, dim=1)

            total_loss = total_loss + F.mse_loss(s_feat, t_feat)

        return total_loss / self.n_levels


class LogitDistillLoss(nn.Module):
    """Logit-based knowledge distillation loss.

    Implements Hinton et al. 2015 ("Distilling the Knowledge in a Neural Network"):
    the student learns to match the teacher's softened output distribution via
    KL divergence on temperature-scaled classification logits, combined with
    an L1 loss on regression outputs to transfer localization knowledge.

    The total distillation loss is:
        L = T^2 * KL(softmax(t_cls/T) || softmax(s_cls/T)) + beta * L1(s_reg, t_reg)

    The T^2 scaling compensates for the gradient magnitude reduction caused by
    temperature scaling (see Hinton et al. 2015, Section 2).

    Args:
        temperature: Softmax temperature for logit softening (default 4.0).
        reg_weight: Weight for regression L1 loss relative to KL loss (default 1.0).
    """

    def __init__(self, temperature=4.0, reg_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.reg_weight = reg_weight

    def forward(self, student_cls, student_reg, teacher_cls, teacher_reg, fg_mask=None):
        """Compute logit distillation loss.

        Args:
            student_cls: (B, N, nc) student classification logits (pre-sigmoid).
            student_reg: (B, N, 4) or (B, N, 4*reg_max) student regression output.
            teacher_cls: (B, N, nc) teacher classification logits (pre-sigmoid).
            teacher_reg: (B, N, 4) or (B, N, 4*reg_max) teacher regression output.
            fg_mask: Optional (B, N) boolean mask. If provided, only compute
                     regression loss on foreground samples.

        Returns:
            Scalar distillation loss.
        """
        T = self.temperature

        # --- Classification KL Divergence ---
        # Temperature-scaled softmax distributions
        # For multi-label classification (sigmoid-based), we treat each class
        # independently using binary KL divergence
        teacher_p = (teacher_cls.detach() / T).sigmoid()
        student_log_p = F.logsigmoid(student_cls / T)
        student_log_one_minus_p = F.logsigmoid(-student_cls / T)

        # Binary KL divergence: sum over classes, mean over batch and anchors
        # KL(teacher || student) = teacher * log(teacher/student) + (1-teacher) * log((1-teacher)/(1-student))
        # Using the stable form with log-sigmoid
        kl_pos = teacher_p * (torch.log(teacher_p.clamp(min=1e-7)) - student_log_p)
        kl_neg = (1.0 - teacher_p) * (
            torch.log((1.0 - teacher_p).clamp(min=1e-7)) - student_log_one_minus_p
        )
        kl_loss = (kl_pos + kl_neg).sum(dim=-1).mean()

        # Scale by T^2 to preserve gradient magnitude (Hinton et al. 2015)
        kl_loss = kl_loss * (T ** 2)

        # --- Regression L1 Loss ---
        t_reg = teacher_reg.detach()
        if fg_mask is not None:
            # Only compute regression loss on foreground anchors
            s_reg_fg = student_reg[fg_mask]
            t_reg_fg = t_reg[fg_mask]
            if s_reg_fg.numel() > 0:
                reg_loss = F.l1_loss(s_reg_fg, t_reg_fg)
            else:
                reg_loss = torch.tensor(0.0, device=student_cls.device)
        else:
            reg_loss = F.l1_loss(student_reg, t_reg)

        return kl_loss + self.reg_weight * reg_loss
