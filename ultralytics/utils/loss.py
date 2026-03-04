# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        self._last_target_scores_sum = float(target_scores_sum)  # expose for CrossKD normalization

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class v8DetectionCrossKDLoss(v8DetectionLoss):
    """Criterion class for computing CrossKD training losses for YOLOv8 object detection.

    Implements true Cross-Head Knowledge Distillation (CrossKD):
      student backbone/neck → student head first layer → [channel adapt] → teacher head remaining layers
                                                                         → cross-head predictions
    The cross-head predictions are then distilled against the teacher's own predictions.
    Gradient flows back through the student head's first layer, giving it a richer signal
    from both the GT loss and the KD loss simultaneously.
    """

    def __init__(self, model, teacher_model, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize v8DetectionCrossKDLoss with student model, teacher model, and KD hyperparameters.

        Args:
            model: Student model (must be de-paralleled).
            teacher_model: Teacher model for knowledge distillation.
            tal_topk: Task-aligned assignment top-k parameter.
            tal_topk2: Task-aligned assignment second top-k parameter.
        """
        super().__init__(model, tal_topk, tal_topk2)
        self._last_target_scores_sum = 1.0  # updated each forward pass via v8DetectionLoss.loss()

        # Store teacher model and extract teacher head
        self.teacher_model = teacher_model
        self.teacher_head = teacher_model.model[-1]  # Detect() module

        # Extract KD hyperparameters
        self.kd_temp = getattr(model.args, "kd_temperature", 4.0)
        self.kd_weight_cls = getattr(model.args, "kd_loss_weight_cls", 1.0)
        self.kd_weight_box = getattr(model.args, "kd_loss_weight_box", 1.0)
        self.kd_freeze_teacher = getattr(model.args, "kd_freeze_teacher", True)

        # Freeze teacher model weights
        if self.kd_freeze_teacher:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        self.teacher_model.eval()

        # Store teacher dtype once to avoid repeated parameter iteration in forward
        self._teacher_dtype = next(teacher_model.parameters()).dtype

        # Validate teacher-student compatibility
        student_head = model.model[-1]

        assert self.teacher_head.nc == student_head.nc, (
            f"Teacher nc={self.teacher_head.nc} must match student nc={student_head.nc}"
        )
        assert self.teacher_head.reg_max == student_head.reg_max, (
            f"Teacher reg_max={self.teacher_head.reg_max} must match student reg_max={student_head.reg_max}"
        )
        assert self.teacher_head.nl == student_head.nl, (
            f"Teacher nl={self.teacher_head.nl} must match student nl={student_head.nl}"
        )

        # Store student head reference for intermediate feature extraction in _forward_cross_head.
        self.student_head = student_head

        # True CrossKD splits each detection head's Sequential at the boundary after its first layer:
        #
        #   cv2[i]:  Conv(ch→c2) | Conv(c2→c2) | Conv2d(c2→4*reg_max)
        #             ^^^student^^   ^^^^^^^^^^^^teacher remaining^^^^^^^^^^^^
        #
        #   cv3[i]:  Sequential(DWConv+Conv(ch→c3)) | Sequential(DWConv+Conv(c3→c3)) | Conv2d(c3→nc)
        #             ^^^^^^^^^^^student^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^teacher remaining^^^^^^^^^^^^
        #
        # adapt_layers_cv2[i]: bridges c2_student → c2_teacher at the cv2 split point.
        # adapt_layers_cv3[i]: bridges c3_student → c3_teacher at the cv3 split point.
        self.adapt_layers_cv2 = nn.ModuleList()
        self.adapt_layers_cv3 = nn.ModuleList()

        for i in range(student_head.nl):
            # --- cv2 (box head) ---
            # student cv2[i][0] is Conv(ch_in, c2_s, 3); its output has c2_s channels.
            # teacher cv2[i][1] is Conv(c2_t, c2_t, 3); its input expects c2_t channels.
            s_c2 = student_head.cv2[i][0].conv.out_channels
            t_c2 = self.teacher_head.cv2[i][1].conv.in_channels
            adapt_cv2 = nn.Conv2d(s_c2, t_c2, 1, bias=False).to(self.device) if s_c2 != t_c2 else nn.Identity()
            if isinstance(adapt_cv2, nn.Conv2d):
                nn.init.kaiming_normal_(adapt_cv2.weight, mode="fan_out")
            self.adapt_layers_cv2.append(adapt_cv2)

            # --- cv3 (cls head) ---
            # legacy   cv3[i][0] = Conv(ch_in, c3, 3)            → output c3 channels
            # non-legacy cv3[i][0] = Sequential(DWConv, Conv(ch_in→c3)) → output c3 channels
            # teacher  cv3[i][1][0] (non-legacy DWConv) or cv3[i][1] (legacy Conv) → input c3_t
            if student_head.legacy:
                s_c3 = student_head.cv3[i][0].conv.out_channels
                t_c3 = self.teacher_head.cv3[i][1].conv.in_channels
            else:
                s_c3 = student_head.cv3[i][0][1].conv.out_channels
                t_c3 = self.teacher_head.cv3[i][1][0].conv.in_channels
            adapt_cv3 = nn.Conv2d(s_c3, t_c3, 1, bias=False).to(self.device) if s_c3 != t_c3 else nn.Identity()
            if isinstance(adapt_cv3, nn.Conv2d):
                nn.init.kaiming_normal_(adapt_cv3.weight, mode="fan_out")
            self.adapt_layers_cv3.append(adapt_cv3)

        # Register both adapt lists on the student model so the optimizer includes their parameters.
        # build_optimizer() must be called after init_criterion() for this to take effect.
        model.kd_adapt_layers = nn.ModuleList([*self.adapt_layers_cv2, *self.adapt_layers_cv3])

    def compute_kd_loss_cls(
        self, cross_head_scores: torch.Tensor, teacher_scores: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification KD loss using Quality Focal Loss (QFL).

        Follows CrossKD paper's design:
        - QFL = |teacher_sigmoid - cross_sigmoid|^beta * BCE, which focuses on positions
          where cross-head and teacher predictions disagree.
        - S(r)=1: all positions contribute equally to the numerator (no complex spatial
          weighting), which is the paper's key claim.
        - Normalized by target_scores_sum (num_pos) to match the GT classification loss
          scale in YOLO. Using batch * num_anchors (paper's |S|) makes the loss orders of
          magnitude smaller because (1) QFL focal weight collapses near 0 for background
          where teacher_probs ≈ cross_probs ≈ 0 due to YOLO's bias_init, and (2) the
          denominator would be ~500x larger than target_scores_sum.
        - sigmoid-based (not softmax) to match YOLO's multi-label binary classification.

        Args:
            cross_head_scores: Cross-head classification predictions (student features → teacher head).
            teacher_scores: Teacher's own classification predictions.

        Returns:
            (torch.Tensor): Weighted classification KD loss.
        """
        T = self.kd_temp
        beta = 1.0  # QFL focusing parameter (same as GFL/CrossKD paper)

        # Teacher sigmoid probabilities as soft targets (with temperature scaling)
        teacher_probs = (teacher_scores / T).sigmoid().detach()

        # Detached cross-head sigmoid for focal weight (no gradient through weight)
        cross_probs = (cross_head_scores / T).sigmoid().detach()

        # QFL focal weight: focus on positions where cross-head and teacher disagree
        focal_weight = (teacher_probs - cross_probs).abs().pow(beta)

        # BCE term (gradient flows through cross_head_scores here)
        bce = F.binary_cross_entropy_with_logits(
            cross_head_scores / T,
            teacher_probs,
            reduction="none",
        )

        # Normalize by target_scores_sum (num_pos) to match GT cls loss scale in YOLO.
        # _last_target_scores_sum is updated by super().loss() before this is called.
        kd_loss = (focal_weight * bce).sum() / self._last_target_scores_sum

        # Scale by T^2 to maintain gradient magnitude
        kd_loss = kd_loss * (T**2)

        # Apply weight
        return kd_loss * self.kd_weight_cls

    def compute_kd_loss_box(
        self, cross_head_boxes: torch.Tensor, teacher_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute bounding box KD loss via Localization Distillation (KL on DFL distributions).

        Args:
            cross_head_boxes: Cross-head box predictions in DFL format (batch, 4*reg_max, num_anchors).
            teacher_boxes: Teacher's own box predictions in DFL format (batch, 4*reg_max, num_anchors).

        Returns:
            (torch.Tensor): Weighted box KD loss.
        """
        # Reshape to (batch*num_anchors*4, reg_max) to treat each coordinate as a distribution
        cross_head_dist = cross_head_boxes.permute(0, 2, 1).reshape(-1, self.reg_max)
        teacher_dist = teacher_boxes.permute(0, 2, 1).reshape(-1, self.reg_max)

        # KL divergence between cross-head and teacher DFL distributions
        kd_loss = F.kl_div(
            F.log_softmax(cross_head_dist, dim=1),
            F.softmax(teacher_dist, dim=1),
            reduction="batchmean",
        )
        return kd_loss * self.kd_weight_box

    def _forward_cross_head(self, student_feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """True CrossKD forward pass: student head first layer → adapt → teacher head remaining layers.

        Data flow per scale i:
            student_feats[i]
              → student cv2[i][0]       (Conv, trained)
              → adapt_layers_cv2[i]     (1×1 conv or Identity, trained)
              → teacher cv2[i][1]       (Conv, frozen)
              → teacher cv2[i][2]       (Conv2d, frozen)
              → cross box predictions

            student_feats[i]
              → student cv3[i][0]       (Sequential DWConv+Conv, trained)
              → adapt_layers_cv3[i]     (1×1 conv or Identity, trained)
              → teacher cv3[i][1]       (Sequential DWConv+Conv, frozen)
              → teacher cv3[i][2]       (Conv2d, frozen)
              → cross cls predictions

        Gradient flows through the student's first layer and the adapt layer,
        giving those weights distillation signal on top of the standard GT loss.

        Args:
            student_feats: Backbone/neck output feature maps, one tensor per detection scale.

        Returns:
            dict with 'boxes' (batch, 4*reg_max, anchors) and 'scores' (batch, nc, anchors).
        """
        bs = student_feats[0].shape[0]
        cross_boxes = []
        cross_scores = []

        for i in range(self.student_head.nl):
            feat = student_feats[i]

            # --- Box head (cv2) ---
            mid_cv2 = self.student_head.cv2[i][0](feat)                    # student layer 0 (grad ON)
            adapt_cv2 = self.adapt_layers_cv2[i]
            if not isinstance(adapt_cv2, nn.Identity):
                mid_cv2 = adapt_cv2(mid_cv2.to(dtype=adapt_cv2.weight.dtype))  # adapt (grad ON)
            mid_cv2 = mid_cv2.to(dtype=self._teacher_dtype)
            out_cv2 = self.teacher_head.cv2[i][1](mid_cv2)                 # teacher layer 1 (frozen)
            out_cv2 = self.teacher_head.cv2[i][2](out_cv2)                 # teacher layer 2 (frozen)
            cross_boxes.append(out_cv2.view(bs, 4 * self.reg_max, -1))

            # --- Class head (cv3) ---
            mid_cv3 = self.student_head.cv3[i][0](feat)                    # student layer 0 (grad ON)
            adapt_cv3 = self.adapt_layers_cv3[i]
            if not isinstance(adapt_cv3, nn.Identity):
                mid_cv3 = adapt_cv3(mid_cv3.to(dtype=adapt_cv3.weight.dtype))  # adapt (grad ON)
            mid_cv3 = mid_cv3.to(dtype=self._teacher_dtype)
            out_cv3 = self.teacher_head.cv3[i][1](mid_cv3)                 # teacher layer 1 (frozen)
            out_cv3 = self.teacher_head.cv3[i][2](out_cv3)                 # teacher layer 2 (frozen)
            cross_scores.append(out_cv3.view(bs, self.nc, -1))

        return {
            "boxes": torch.cat(cross_boxes, dim=-1),
            "scores": torch.cat(cross_scores, dim=-1),
        }

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate combined ground-truth and knowledge distillation losses.

        Args:
            preds: Student model predictions containing 'boxes', 'scores', and 'feats'.
            batch: Batch data containing images and ground-truth annotations.

        Returns:
            (tuple): Total loss and detached loss items for logging.
        """
        # 1. Compute standard ground-truth loss
        gt_loss, gt_loss_detach = super().loss(preds, batch)

        # 2. Get teacher predictions with no gradient
        with torch.no_grad():
            self.teacher_model.eval()  # Ensure teacher is in eval mode
            teacher_input = batch["img"].to(dtype=self._teacher_dtype)
            teacher_output = self.teacher_model(teacher_input)

            # Extract predictions dict from output
            if isinstance(teacher_output, tuple):
                teacher_preds = teacher_output[1]  # (y, preds) format
            else:
                teacher_preds = teacher_output

            # Handle end2end models
            if isinstance(teacher_preds, dict) and "one2many" in teacher_preds:
                teacher_preds = teacher_preds["one2many"]

        # 3. True CrossKD: student feat → student head[0] → adapt → teacher head[1:] → cross predictions
        student_feats = preds["feats"]
        cross_head_preds = self._forward_cross_head(student_feats)

        # 4. Compute KD losses
        kd_loss_cls = self.compute_kd_loss_cls(cross_head_preds["scores"], teacher_preds["scores"])

        kd_loss_box = self.compute_kd_loss_box(cross_head_preds["boxes"], teacher_preds["boxes"])

        # 5. Combine losses
        # gt_loss is a 3-element tensor [box*bs, cls*bs, dfl*bs].
        # Appending KD losses as separate elements ensures loss.sum() counts each exactly once.
        # KD losses are multiplied by batch_size to match the scaling of gt_loss.
        batch_size = preds["boxes"].shape[0]
        total_loss = torch.cat([
            gt_loss,
            (kd_loss_cls * batch_size).reshape(1),
            (kd_loss_box * batch_size).reshape(1),
        ])
        
        # 6. Prepare detached losses for logging (box, cls, dfl, kd_cls, kd_box)
        total_loss_detach = torch.cat(
            [gt_loss_detach, torch.tensor([kd_loss_cls.item(), kd_loss_box.item()], device=self.device)]
        )

        return total_loss, total_loss_detach


class FGDScaleLoss(nn.Module):
    """Single-scale FGD loss module for one FPN level.

    Computes four sub-losses between student and teacher feature maps:
      - fg_loss:   MSE in foreground regions (attention-weighted)
      - bg_loss:   MSE in background regions (attention-weighted)
      - mask_loss: L1 distance between spatial/channel attention maps
      - rela_loss: GCNet-style context-augmented MSE
    """

    def __init__(self, student_ch: int, teacher_ch: int, temp: float = 0.5):
        """Initialize FGDScaleLoss.

        Args:
            student_ch: Number of student feature channels.
            teacher_ch: Number of teacher feature channels.
            temp: Softmax temperature for attention computation.
        """
        super().__init__()
        from ultralytics.nn.modules import LayerNorm2d

        self.temp = temp

        # Channel alignment: project student to teacher channel count
        self.align = (
            nn.Conv2d(student_ch, teacher_ch, 1, bias=False)
            if student_ch != teacher_ch
            else nn.Identity()
        )
        if isinstance(self.align, nn.Conv2d):
            nn.init.kaiming_normal_(self.align.weight, mode="fan_out")

        # Spatial attention: 1×1 conv, kaiming-initialized (matches official FGD)
        self.conv_mask_s = nn.Conv2d(teacher_ch, 1, 1)
        self.conv_mask_t = nn.Conv2d(teacher_ch, 1, 1)
        nn.init.kaiming_normal_(self.conv_mask_s.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.conv_mask_t.weight, mode="fan_in")

        # Channel-add context branch (GCNet style), operates on teacher_ch channels
        # Last conv is zero-initialized so the branch starts as identity (matches official FGD)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_ch, teacher_ch // 2, 1),
            LayerNorm2d(teacher_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_ch // 2, teacher_ch, 1),
        )
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_ch, teacher_ch // 2, 1),
            LayerNorm2d(teacher_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_ch // 2, teacher_ch, 1),
        )
        nn.init.zeros_(self.channel_add_conv_s[-1].weight)
        nn.init.zeros_(self.channel_add_conv_s[-1].bias)
        nn.init.zeros_(self.channel_add_conv_t[-1].weight)
        nn.init.zeros_(self.channel_add_conv_t[-1].bias)

    def get_attention(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial and channel attention maps.

        Args:
            feat: Feature map of shape (B, C, H, W).

        Returns:
            S_attn: Spatial attention (B, 1, H, W), softmax normalized × H*W.
            C_attn: Channel attention (B, C, 1, 1), softmax normalized × C.
        """
        B, C, H, W = feat.shape
        # Spatial attention: mean over channels, then softmax
        s = feat.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        s = (s / self.temp).reshape(B, -1)         # (B, H*W)
        s = s.softmax(dim=-1) * (H * W)            # normalize: sum = H*W
        S_attn = s.reshape(B, 1, H, W)

        # Channel attention: mean over spatial, then softmax
        c = feat.abs().mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        c = (c / self.temp).reshape(B, -1)              # (B, C)
        c = c.softmax(dim=-1) * C
        C_attn = c.reshape(B, C, 1, 1)

        return S_attn, C_attn

    def spatial_pool(self, feat: torch.Tensor, conv_mask: nn.Module) -> torch.Tensor:
        """GCNet-style spatial pooling to produce a context vector.

        Args:
            feat: Feature map (B, C, H, W).
            conv_mask: 1×1 conv producing attention logits (B, 1, H, W).

        Returns:
            context: (B, C, 1, 1)
        """
        B, C, H, W = feat.shape
        attn = conv_mask(feat).reshape(B, 1, H * W)  # (B, 1, H*W)
        attn = attn.softmax(dim=-1)                   # (B, 1, H*W)
        feat_flat = feat.reshape(B, C, H * W)         # (B, C, H*W)
        context = torch.bmm(feat_flat, attn.transpose(1, 2))  # (B, C, 1)
        return context.reshape(B, C, 1, 1)

    def get_fea_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        Mask_fg: torch.Tensor,
        Mask_bg: torch.Tensor,
        C_s: torch.Tensor,
        C_t: torch.Tensor,
        S_s: torch.Tensor,
        S_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute foreground and background feature loss.

        Args:
            preds_S: Aligned student features (B, C_t, H, W).
            preds_T: Teacher features (B, C_t, H, W).
            Mask_fg: Foreground mask (B, 1, H, W).
            Mask_bg: Background mask (B, 1, H, W).
            C_s, C_t: Channel attention maps (B, C_t, 1, 1).
            S_s, S_t: Spatial attention maps (B, 1, H, W).

        Returns:
            (fg_loss, bg_loss): Scalar tensors.
        """
        # Weight: sqrt of teacher spatial × sqrt of teacher channel attention
        weight = S_t.sqrt() * C_t.sqrt()  # (B, C_t, H, W) broadcast

        B = preds_T.shape[0]

        # Foreground loss: sum / B (matches official FGD)
        diff_fg = (preds_S - preds_T) * weight * Mask_fg.sqrt()
        fg_loss = (diff_fg ** 2).sum() / B

        # Background loss: sum / B (matches official FGD)
        diff_bg = (preds_S - preds_T) * weight * Mask_bg.sqrt()
        bg_loss = (diff_bg ** 2).sum() / B

        return fg_loss, bg_loss

    def get_mask_loss(
        self,
        C_s: torch.Tensor,
        C_t: torch.Tensor,
        S_s: torch.Tensor,
        S_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L1 attention mask distillation loss.

        Args:
            C_s, C_t: Channel attention maps (B, C_t, 1, 1).
            S_s, S_t: Spatial attention maps (B, 1, H, W).

        Returns:
            mask_loss: Scalar tensor.
        """
        B = C_s.shape[0]
        # sum / B for each term (matches official FGD: torch.sum(|diff|) / len(x))
        return (C_s - C_t).abs().sum() / B + (S_s - S_t).abs().sum() / B

    def get_rela_loss(self, preds_S: torch.Tensor, preds_T: torch.Tensor) -> torch.Tensor:
        """Compute GCNet-style relational context loss.

        Args:
            preds_S: Aligned student features (B, C_t, H, W).
            preds_T: Teacher features (B, C_t, H, W).

        Returns:
            rela_loss: Scalar tensor.
        """
        context_S = self.spatial_pool(preds_S, self.conv_mask_s)  # (B, C_t, 1, 1)
        context_T = self.spatial_pool(preds_T, self.conv_mask_t)

        aug_S = preds_S + self.channel_add_conv_s(context_S)
        aug_T = preds_T + self.channel_add_conv_t(context_T)

        B = aug_S.shape[0]
        # sum / B (matches official FGD: MSELoss(reduction='sum') / len(out_s))
        return ((aug_S - aug_T) ** 2).sum() / B

    def forward(
        self,
        feat_S: torch.Tensor,
        feat_T: torch.Tensor,
        gt_bboxes: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute FGD sub-losses for one FPN scale.

        Args:
            feat_S: Student feature map (B, C_s, H, W).
            feat_T: Teacher feature map (B, C_t, H, W), detached externally.
            gt_bboxes: List of GT boxes per image in feature-map-space xyxy (list[Tensor(N,4)]).

        Returns:
            (fg_loss, bg_loss, mask_loss, rela_loss): Scalar tensors.
        """
        B, _, H, W = feat_T.shape

        # Cast inputs to the module's parameter dtype (fp32) to avoid mixed-precision conflicts
        module_dtype = next(self.parameters()).dtype
        feat_S = feat_S.to(dtype=module_dtype)
        feat_T = feat_T.to(dtype=module_dtype)

        # Align student channels to teacher channels
        preds_S = self.align(feat_S)
        preds_T = feat_T  # already detached by caller

        # Compute attention maps
        S_s, C_s = self.get_attention(preds_S)
        S_t, C_t = self.get_attention(preds_T)

        # Build fg/bg masks from GT boxes
        Mask_fg = torch.zeros(B, 1, H, W, device=feat_S.device, dtype=module_dtype)
        for b in range(B):
            boxes = gt_bboxes[b]  # (N, 4) in feat-space xyxy
            if boxes.numel() == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.clamp(min=0)
                x1i, y1i = int(x1.floor()), int(y1.floor())
                x2i, y2i = int(x2.ceil()), int(y2.ceil())
                x2i, y2i = min(x2i, W), min(y2i, H)
                if x2i > x1i and y2i > y1i:
                    area = (x2i - x1i) * (y2i - y1i)
                    val = torch.tensor(1.0 / area, dtype=module_dtype, device=feat_S.device)
                    Mask_fg[b, 0, y1i:y2i, x1i:x2i] = torch.maximum(
                        Mask_fg[b, 0, y1i:y2i, x1i:x2i], val
                    )

        # Derive bg mask: strictly binary (0 where fg > 0, 1 elsewhere), then normalize
        # by background pixel count per image — matches official FGD implementation.
        Mask_bg = torch.zeros_like(Mask_fg)
        for b in range(B):
            Mask_bg[b] = torch.where(Mask_fg[b] > 0, torch.zeros_like(Mask_fg[b]), torch.ones_like(Mask_fg[b]))
            bg_sum = Mask_bg[b].sum()
            if bg_sum > 0:
                Mask_bg[b] = Mask_bg[b] / bg_sum

        # Compute sub-losses
        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t)
        mask_loss = self.get_mask_loss(C_s, C_t, S_s, S_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)

        return fg_loss, bg_loss, mask_loss, rela_loss


class v8DetectionFGDLoss(v8DetectionLoss):
    """Criterion class for computing FGD training losses for YOLOv8 object detection.

    Implements Focal and Global Knowledge Distillation (FGD):
      Distills intermediate backbone/neck feature maps at each FPN scale using
      foreground-focused attention and global relational context losses.
    """

    def __init__(self, model, teacher_model):
        """Initialize v8DetectionFGDLoss.

        Args:
            model: Student model (de-paralleled).
            teacher_model: Teacher model for KD.
        """
        super().__init__(model, tal_topk=10)

        # Hyperparameters
        self.fgd_temp = getattr(model.args, "fgd_temp", 0.5)
        self.fgd_alpha = getattr(model.args, "fgd_alpha", 0.001)
        self.fgd_beta = getattr(model.args, "fgd_beta", 0.0005)
        self.fgd_gamma = getattr(model.args, "fgd_gamma", 0.001)
        self.fgd_lambda = getattr(model.args, "fgd_lambda", 0.000005)

        self.teacher_model = teacher_model
        kd_freeze_teacher = getattr(model.args, "kd_freeze_teacher", True)
        if kd_freeze_teacher:
            for param in teacher_model.parameters():
                param.requires_grad = False
        self.teacher_model.eval()
        self._teacher_dtype = next(teacher_model.parameters()).dtype

        # Extract FPN channel counts from Detect head
        student_head = model.model[-1]
        teacher_head = teacher_model.model[-1]
        nl = student_head.nl

        student_channels = [student_head.cv2[i][0].conv.in_channels for i in range(nl)]
        teacher_channels = [teacher_head.cv2[i][0].conv.in_channels for i in range(nl)]

        # One FGDScaleLoss per FPN level — move to student device immediately (same pattern as CrossKD adapt layers)
        self.fgd_losses = nn.ModuleList([
            FGDScaleLoss(student_channels[i], teacher_channels[i], self.fgd_temp).to(self.device)
            for i in range(nl)
        ])

        # Register on model so optimizer picks up fgd_losses parameters
        model.fgd_modules = self.fgd_losses

        self._nl = nl

    def _make_gt_bboxes_for_scale(
        self,
        batch: dict,
        feat_h: int,
        feat_w: int,
        img_h: int,
        img_w: int,
    ) -> list[torch.Tensor]:
        """Convert batch GT boxes to feature-map-space xyxy per image.

        Args:
            batch: Training batch dict with 'bboxes' (normalized xywh) and 'batch_idx'.
            feat_h, feat_w: Feature map spatial dimensions.
            img_h, img_w: Input image spatial dimensions.

        Returns:
            List of length B, each element Tensor(N_i, 4) in feat-space xyxy.
        """
        bboxes_norm = batch["bboxes"]  # (total_instances, 4) normalized xywh
        batch_idx = batch["batch_idx"].long()
        B = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else batch["img"].shape[0]

        # Convert normalized xywh → pixel xyxy in feature-map space
        scale_x = feat_w / img_w
        scale_y = feat_h / img_h

        cx = bboxes_norm[:, 0] * img_w * scale_x
        cy = bboxes_norm[:, 1] * img_h * scale_y
        bw = bboxes_norm[:, 2] * img_w * scale_x
        bh = bboxes_norm[:, 3] * img_h * scale_y

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        boxes_feat = torch.stack([x1, y1, x2, y2], dim=1)  # (total, 4)

        result = []
        for b in range(B):
            mask = batch_idx == b
            result.append(boxes_feat[mask])
        return result

    def loss(
        self, preds: dict, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute combined GT + FGD losses.

        Args:
            preds: Student predictions dict with 'feats' key.
            batch: Training batch dict.

        Returns:
            (total_loss, total_loss_detach): 7-element tensors.
        """
        # 1. Standard GT loss
        gt_loss, gt_loss_detach = super().loss(preds, batch)

        # 2. Teacher forward (no grad)
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_output = self.teacher_model(batch["img"].to(dtype=self._teacher_dtype))
            if isinstance(teacher_output, tuple):
                teacher_preds = teacher_output[1]
            else:
                teacher_preds = teacher_output
            if isinstance(teacher_preds, dict) and "one2many" in teacher_preds:
                teacher_preds = teacher_preds["one2many"]
            teacher_feats = teacher_preds["feats"]

        # 3. Student FPN features
        student_feats = preds["feats"]
        

        img_h, img_w = batch["img"].shape[2], batch["img"].shape[3]
        batch_size = batch["img"].shape[0]

        # 4. Per-scale FGD losses
        total_fg = torch.zeros(1, device=self.device)
        total_bg = torch.zeros(1, device=self.device)
        total_mask = torch.zeros(1, device=self.device)
        total_rela = torch.zeros(1, device=self.device)

        for i in range(self._nl):
            s_feat = student_feats[i]
            t_feat = teacher_feats[i].detach().to(dtype=s_feat.dtype)

            assert s_feat.shape[2:] == t_feat.shape[2:], (
                f"FGD scale {i}: student spatial {s_feat.shape[2:]} != teacher {t_feat.shape[2:]}"
            )

            feat_h, feat_w = s_feat.shape[2], s_feat.shape[3]
            gt_bboxes = self._make_gt_bboxes_for_scale(batch, feat_h, feat_w, img_h, img_w)

            fg, bg, mask, rela = self.fgd_losses[i](s_feat, t_feat, gt_bboxes)
            total_fg = total_fg + fg
            total_bg = total_bg + bg
            total_mask = total_mask + mask
            total_rela = total_rela + rela

        # Apply weights and scale by batch_size (match GT loss scaling convention)
        # Scales are summed (not averaged) to match official FGD hook behaviour.
        fg_loss = self.fgd_alpha * total_fg * batch_size
        bg_loss = self.fgd_beta * total_bg * batch_size
        mask_loss = self.fgd_gamma * total_mask * batch_size
        rela_loss = self.fgd_lambda * total_rela * batch_size

        # 5. Combine [box, cls, dfl, fgd_fg, fgd_bg, fgd_mask, fgd_rela]
        total_loss = torch.cat([
            gt_loss,
            fg_loss.reshape(1),
            bg_loss.reshape(1),
            mask_loss.reshape(1),
            rela_loss.reshape(1),
        ])

        # Display weighted per-sample values (consistent with GT loss scale).
        total_loss_detach = torch.cat([
            gt_loss_detach,
            torch.cat([fg_loss, bg_loss, mask_loss, rela_loss]).detach() / batch_size,
        ])

        return total_loss, total_loss_detach


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semseg
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semseg)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
