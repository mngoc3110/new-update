# utils/loss.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal


def _ramp_weight(epoch: int, warmup: int, ramp: int, max_w: float) -> float:
    """
    epoch: 0-based
    warmup: số epoch đầu weight=0
    ramp: số epoch tăng dần từ 0 -> max_w
    """
    if max_w == 0:
        return 0.0
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return float(max_w)
    t = (epoch - warmup) / float(ramp)  # 0..1
    t = max(0.0, min(1.0, t))
    return float(max_w) * t

# --- Baseline Loss Classes ---
class LSR2(nn.Module):
    def __init__(self, e=0.1, label_mode='soft'): # Added default args for compatibility
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.label_mode = label_mode

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        mask = (one_hot==0)
        # Hardcoded balance weights from Baseline
        balance_weight = torch.tensor([0.065267810,0.817977729,1.035884371,0.388144355,0.19551041668]).to(one_hot.device)
        ex_weight = balance_weight.expand(one_hot.size(0),-1)
        resize_weight = ex_weight[mask].view(one_hot.size(0),-1)
        resize_weight /= resize_weight.sum(dim=1, keepdim=True)
        one_hot[mask] += (resize_weight*smooth_factor).view(-1)
        return one_hot.to(target.device)
    
    def forward(self, x, target):
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)

class BlvLoss(nn.Module):
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_list)) - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name

    def forward(self, pred, target):
        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)
        pred = pred + (viariation.abs() / self.frequency_list.max() * self.frequency_list)
        loss = F.cross_entropy(pred, target, reduction='none')

        return loss.mean()

# --- Existing Loss Classes ---

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if alpha is not None:
            # Register as buffer to ensure it moves to device automatically
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # inputs: logits [B, C]
        # targets: labels [B]
        
        # Calculate CE loss without direct 'weight' parameter to avoid MPS error
        ce_loss_unweighted = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Manually apply class weights if provided
        if self.alpha is not None:
            # Ensure alpha is on the correct device (fallback safety)
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
                
            # self.alpha is class_weights, shape (C,)
            # targets are class indices, shape (N,)
            # lookup weights for each target sample
            sample_weights = self.alpha[targets] 
            ce_loss = ce_loss_unweighted * sample_weights
        else:
            ce_loss = ce_loss_unweighted

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CLIPCAERLoss(nn.Module):
    """
    L = CE/Focal + w_mi(epoch)*MI + w_dc(epoch)*DC + w_cons*Consistency
    - CE: optional label smoothing OR semantic label smoothing (LDLVA-inspired)
    - MI: InfoNCE-ish using your mi_estimator(pos-neg)
    - DC: KL( P_joint || P_l ⊗ P_h ) theo paper, tính từ logits 2 view
    - Cons: KL( Learnable || Hand-crafted ) để giữ kiến thức gốc
    """

    def __init__(self, args, mi_estimator=None, num_classes=5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.mi_estimator = mi_estimator
        self.args = args

        # base lambdas (max weight)
        self.lambda_mi = float(getattr(args, "lambda_mi", 1.0))
        self.lambda_dc = float(getattr(args, "lambda_dc", 0.0))
        self.lambda_cons = float(getattr(args, "lambda_cons", 0.0))
        self.lambda_binary = float(getattr(args, "lambda_binary", 0.3))
        
        # warmup/ramp
        self.mi_warmup = int(getattr(args, "mi_warmup", 0))
        self.mi_ramp   = int(getattr(args, "mi_ramp", 0))
        self.dc_warmup = int(getattr(args, "dc_warmup", 0))
        self.dc_ramp   = int(getattr(args, "dc_ramp", 0))

        # Logit Adjustment (Menon et al., 2020)
        self.register_buffer("class_priors", None)
        if hasattr(args, "class_counts") and args.class_counts is not None:
            counts = torch.tensor(args.class_counts, dtype=torch.float32)
            priors = counts / counts.sum()
            self.register_buffer("class_priors", priors)
        
        # Base class weights for the MAIN 5-class loss
        cw = getattr(args, "class_weights", None)
        if cw is not None:
            cw = torch.tensor(cw, dtype=torch.float32)
        self.register_buffer("base_class_weights", cw)

        # Loss for the Binary Head
        self.binary_loss = nn.CrossEntropyLoss()

        # cache weights for printing
        self.last_w_mi = 0.0
        self.last_w_dc = 0.0
        
        # Cache for prior distribution matrix
        self.prior_distribution = None

    def set_epoch(self, epoch: int):
        """Called every epoch to update MI/DC weights."""
        w_mi = _ramp_weight(epoch, self.mi_warmup, self.mi_ramp, self.lambda_mi)
        w_dc = _ramp_weight(epoch, self.dc_warmup, self.dc_ramp, self.lambda_dc)
        self.last_w_mi = w_mi
        self.last_w_dc = w_dc

    def _get_stage_params(self, epoch):
        """Determine current stage and return its parameters."""
        stage1_end = self.args.stage1_epochs
        stage2_end = stage1_end + self.args.stage2_epochs
        stage3_end = stage2_end + self.args.stage3_epochs

        if epoch < stage1_end:
            stage = 1
            params = {
                "max_class_weight": -1, # No clipping
                "logit_adjust_tau": 0.0,
                "use_focal_loss": False,
                "label_smoothing": self.args.stage1_label_smoothing,
                "semantic_smoothing": True,
                "smoothing_temp": self.args.stage1_smoothing_temp,
            }
        elif epoch < stage2_end:
            stage = 2
            params = {
                "max_class_weight": self.args.stage2_max_class_weight,
                "logit_adjust_tau": self.args.stage2_logit_adjust_tau,
                "use_focal_loss": True,
                "focal_gamma": self.args.focal_gamma,
                "label_smoothing": self.args.stage2_label_smoothing,
                "semantic_smoothing": True,
                "smoothing_temp": self.args.stage2_smoothing_temp,
            }
        elif epoch < stage3_end:
            stage = 3
            params = {
                "max_class_weight": self.args.stage3_max_class_weight,
                "logit_adjust_tau": self.args.stage3_logit_adjust_tau,
                "use_focal_loss": True,
                "focal_gamma": self.args.focal_gamma,
                "label_smoothing": self.args.label_smoothing,
                "semantic_smoothing": self.args.semantic_smoothing,
                "smoothing_temp": self.args.stage3_smoothing_temp,
            }
        else:
            stage = 4
            params = {
                "max_class_weight": self.args.stage4_max_class_weight,
                "logit_adjust_tau": self.args.stage4_logit_adjust_tau,
                "use_focal_loss": getattr(self.args, 'stage4_use_focal_loss', 'True') == 'True',
                "focal_gamma": getattr(self.args, 'stage4_focal_gamma', self.args.focal_gamma),
                "label_smoothing": self.args.label_smoothing,
                "semantic_smoothing": getattr(self.args, 'stage4_semantic_smoothing', 'True') == 'True',
                "smoothing_temp": getattr(self.args, 'stage4_smoothing_temp', self.args.smoothing_temp),
            }
        
        # Special handling for Neutral in logit adjustment
        params["logit_adjust_tau_neutral"] = getattr(self.args, f'stage{stage}_logit_adjust_tau_neutral', params["logit_adjust_tau"])
        
        # Get class weights for the current stage
        current_weights = self.base_class_weights.clone() if self.base_class_weights is not None else None
        if current_weights is not None and params["max_class_weight"] > 0:
            current_weights = torch.clamp(current_weights, max=params["max_class_weight"])
        
        # Neutral-preserving weight adjustment for Stage 4
        if stage == 4 and hasattr(self.args, 'stage4_neutral_weight') and self.args.stage4_neutral_weight > 0 and current_weights is not None:
            current_weights[0] = self.args.stage4_neutral_weight
        
        params["class_weights"] = current_weights
        
        return params


    def _sanitize_targets(self, targets):
        if targets.dim() > 1:
            targets = targets.view(-1)
        return targets.long().clamp(0, self.num_classes - 1)

    def _mi_loss(self, f_l, f_h):
        if (f_l is None or f_h is None or self.mi_estimator is None or self.last_w_mi == 0.0):
            return torch.tensor(0.0, device=f_l.device if isinstance(f_l, torch.Tensor) else 'cpu')
        pos = self.mi_estimator(f_l.float(), f_h.float()).mean()
        neg = self.mi_estimator(f_l.float(), f_h.float()[torch.randperm(f_h.size(0))]).mean()
        return -(pos - neg)

    def _dc_loss(self, logits_l, logits_h, eps=1e-8):
        if logits_l is None or logits_h is None or self.last_w_dc == 0.0:
            return torch.tensor(0.0, device=logits_l.device if isinstance(logits_l, torch.Tensor) else 'cpu')
        p_l = F.softmax(logits_l.float(), dim=1)
        p_h = F.softmax(logits_h.float(), dim=1)
        P = (p_l.unsqueeze(2) * p_h.unsqueeze(1)).sum(0)
        P = P / (P.sum() + eps)
        P_l = P.sum(dim=1, keepdim=True).clamp_min(eps)
        P_h = P.sum(dim=0, keepdim=True).clamp_min(eps)
        P = P.clamp_min(eps)
        dc = (P * (torch.log(P) - torch.log(P_l) - torch.log(P_h))).sum()
        return dc
    
    def _consistency_loss(self, logits_learnable, logits_frozen, T=1.0):
        if logits_frozen is None or self.lambda_cons == 0.0:
            return torch.tensor(0.0, device=logits_learnable.device)
        p_frozen = F.softmax(logits_frozen / T, dim=1)
        log_p_learnable = F.log_softmax(logits_learnable / T, dim=1)
        loss = F.kl_div(log_p_learnable, p_frozen, reduction='batchmean') * (T**2)
        return loss

    def _compute_semantic_target(self, targets, hand_crafted_text_features, smoothing_temp, label_smoothing):
        if self.prior_distribution is None or self.prior_distribution.device != hand_crafted_text_features.device:
            sim_matrix = hand_crafted_text_features @ hand_crafted_text_features.t()
            self.prior_distribution = F.softmax(sim_matrix / smoothing_temp, dim=1)
        batch_prior = self.prior_distribution[targets]
        one_hot = F.one_hot(targets, self.num_classes).float()
        soft_targets = (1.0 - label_smoothing) * one_hot + label_smoothing * batch_prior
        return soft_targets

    def forward(
        self,
        logits,
        targets,
        *,
        epoch: int = 0, # Default to 0 for inference
        learnable_text_features=None,
        hand_crafted_text_features=None,
        logits_hand=None,
        logits_binary=None,
        is_train=True,
    ):
        if is_train and epoch is not None:
            self.set_epoch(int(epoch))
        
        # Get stage-specific parameters
        stage_params = self._get_stage_params(epoch if is_train else self.args.epochs) # Use last stage for eval
        
        # Apply logit adjustment
        if self.class_priors is not None and stage_params["logit_adjust_tau"] > 0:
            tau = torch.full((self.num_classes,), stage_params["logit_adjust_tau"], device=logits.device)
            tau[0] = stage_params["logit_adjust_tau_neutral"] # Special tau for Neutral (class 0)
            logits = logits - tau * torch.log(self.class_priors + 1e-9)

        # Apply post-hoc inference bias
        if not is_train and hasattr(self.args, 'inference_neutral_bias') and self.args.inference_neutral_bias != 0.0:
            logits[:, 0] += self.args.inference_neutral_bias

        targets_main = self._sanitize_targets(targets)
        
        # --- Main 5-class loss ---
        current_weights = stage_params["class_weights"]
        if stage_params["semantic_smoothing"] and stage_params["label_smoothing"] > 0.0 and hand_crafted_text_features is not None:
            soft_targets = self._compute_semantic_target(targets_main, hand_crafted_text_features.float(), stage_params["smoothing_temp"], stage_params["label_smoothing"])
            log_probs = F.log_softmax(logits, dim=1)
            per_sample_loss = -torch.sum(soft_targets * log_probs, dim=1)
            if current_weights is not None:
                sample_weights = current_weights.to(targets_main.device)[targets_main]
                ce_main_loss = (per_sample_loss * sample_weights).mean()
            else:
                ce_main_loss = per_sample_loss.mean()
        else:
            if stage_params["use_focal_loss"]:
                loss_func = FocalLoss(gamma=stage_params["focal_gamma"], alpha=current_weights, label_smoothing=stage_params["label_smoothing"])
            else:
                loss_func = nn.CrossEntropyLoss(weight=current_weights.to(logits.device) if current_weights is not None else None, label_smoothing=stage_params["label_smoothing"])
            ce_main_loss = loss_func(logits, targets_main)

        # --- Other loss components ---
        mi = self._mi_loss(learnable_text_features, hand_crafted_text_features)
        dc = self._dc_loss(logits, logits_hand)
        cons_loss = self._consistency_loss(logits, logits_hand)
        
        binary_loss_component = torch.tensor(0.0, device=logits.device)
        if logits_binary is not None and self.lambda_binary > 0:
            targets_binary = (targets > 0).long()
            binary_loss_component = self.binary_loss(logits_binary, targets_binary)

        total = ce_main_loss + self.last_w_mi * mi + self.last_w_dc * dc + self.lambda_cons * cons_loss + self.lambda_binary * binary_loss_component

        return {
            "total": total, "ce": ce_main_loss, "mi": mi, "dc": dc,
            "cons": cons_loss, "binary": binary_loss_component,
            "w_mi": float(self.last_w_mi), "w_dc": float(self.last_w_dc),
        }


def build_criterion(args, mi_estimator=None, num_classes=5):
    return CLIPCAERLoss(args, mi_estimator=mi_estimator, num_classes=num_classes)