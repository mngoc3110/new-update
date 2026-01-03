# utils/loss.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Alpha can be class weights
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs: logits [B, C]
        # targets: labels [B]
        
        # Apply label smoothing manually if needed, but Focal Loss typically works with hard labels.
        # For simplicity and standard Focal Loss, we use hard labels here or smoothed one-hot.
        # However, to keep it compatible with standard CE interface:
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, label_smoothing=self.label_smoothing)
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
    L = CE/Focal + w_mi(epoch)*MI + w_dc(epoch)*DC
    - CE: optional label smoothing OR semantic label smoothing (LDLVA-inspired)
    - MI: InfoNCE-ish using your mi_estimator(pos-neg)
    - DC: KL( P_joint || P_l ⊗ P_h ) theo paper, tính từ logits 2 view
    """

    def __init__(self, args, mi_estimator=None, num_classes=5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.mi_estimator = mi_estimator

        # base lambdas (max weight)
        self.lambda_mi = float(getattr(args, "lambda_mi", 1.0))
        self.lambda_dc = float(getattr(args, "lambda_dc", 0.0))

        # warmup/ramp
        self.mi_warmup = int(getattr(args, "mi_warmup", 0))
        self.mi_ramp   = int(getattr(args, "mi_ramp", 0))
        self.dc_warmup = int(getattr(args, "dc_warmup", 0))
        self.dc_ramp   = int(getattr(args, "dc_ramp", 0))

        # label smoothing
        self.label_smoothing = float(getattr(args, "label_smoothing", 0.0))
        self.semantic_smoothing = (str(getattr(args, "semantic_smoothing", "False")) == "True")
        self.smoothing_temp = float(getattr(args, "smoothing_temp", 0.1))

        # Logit Adjustment (Menon et al., 2020)
        self.logit_adjust_tau = float(getattr(args, "logit_adjust_tau", 0.0))
        self.register_buffer("class_priors", None)
        if hasattr(args, "class_counts") and args.class_counts is not None:
            counts = torch.tensor(args.class_counts, dtype=torch.float32)
            priors = counts / counts.sum()
            self.register_buffer("class_priors", priors)

        # optional class weights (nếu bạn muốn dùng)
        cw = getattr(args, "class_weights", None)  # list/tuple or None
        if cw is not None:
            cw = torch.tensor(cw, dtype=torch.float32)
        self.register_buffer("class_weights", cw if cw is not None else None)
        
        # Focal Loss
        self.use_focal_loss = (str(getattr(args, "use_focal_loss", "False")) == "True")
        self.focal_gamma = float(getattr(args, "focal_gamma", 2.0))

        if self.semantic_smoothing and self.label_smoothing > 0.0:
            # If Semantic Smoothing is ON, we use KLDivLoss manually
            self.ce_loss = None 
        else:
            if self.use_focal_loss:
                # Use Focal Loss
                self.ce_loss = FocalLoss(
                    gamma=self.focal_gamma,
                    alpha=self.class_weights,
                    label_smoothing=self.label_smoothing
                )
            else:
                # Standard CE (with or without uniform smoothing)
                self.ce_loss = nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    label_smoothing=self.label_smoothing,
                )

        # cache weights for printing (trainer có thể đọc)
        self.last_w_mi = 0.0
        self.last_w_dc = 0.0
        
        # Cache for prior distribution matrix
        self.prior_distribution = None

    def set_epoch(self, epoch: int):
        """Gọi mỗi epoch để cập nhật weight MI/DC."""
        w_mi = _ramp_weight(epoch, self.mi_warmup, self.mi_ramp, self.lambda_mi)
        w_dc = _ramp_weight(epoch, self.dc_warmup, self.dc_ramp, self.lambda_dc)
        self.last_w_mi = w_mi
        self.last_w_dc = w_dc

    # ---------------- utils ----------------
    def _sanitize_targets(self, targets):
        if targets.dim() > 1:
            targets = targets.view(-1)
        return targets.long().clamp(0, self.num_classes - 1)

    # ---------------- MI ----------------
    def _mi_loss(self, f_l, f_h):
        """
        f_l: learnable_text_features [C,D] hoặc [B,D]
        f_h: handcrafted_text_features [C,D] hoặc [B,D]
        Với code hiện tại của bạn: text_features thường [C,D]
        -> MI ở mức "set-wise" vẫn chạy, nhưng ổn định hơn nếu float32.
        """
        if (
            f_l is None
            or f_h is None
            or self.mi_estimator is None
            or self.last_w_mi == 0.0
        ):
            # trả về 0 đúng dtype/device
            if isinstance(f_l, torch.Tensor):
                return f_l.new_tensor(0.0)
            if isinstance(f_h, torch.Tensor):
                return f_h.new_tensor(0.0)
            return torch.tensor(0.0)

        f_l = f_l.float()
        f_h = f_h.float()

        pos = self.mi_estimator(f_l, f_h).mean()
        idx = torch.randperm(f_h.size(0), device=f_h.device)
        neg = self.mi_estimator(f_l, f_h[idx]).mean()

        return -(pos - neg)

    # ---------------- DC ----------------
    def _dc_loss(self, logits_l, logits_h, eps=1e-8):
        """
        DC theo paper (ổn định):
          p_l = softmax(logits_l)
          p_h = softmax(logits_h)
          P = sum_b p_l(b) ⊗ p_h(b)  (joint)
          dc = KL( P || P_l ⊗ P_h )
        """
        if logits_l is None or logits_h is None or self.last_w_dc == 0.0:
            if isinstance(logits_l, torch.Tensor):
                return logits_l.new_tensor(0.0)
            if isinstance(logits_h, torch.Tensor):
                return logits_h.new_tensor(0.0)
            return torch.tensor(0.0)

        p_l = F.softmax(logits_l.float(), dim=1)
        p_h = F.softmax(logits_h.float(), dim=1)

        P = torch.einsum("bi,bj->ij", p_l, p_h)
        P = P / (P.sum() + eps)

        P_l = P.sum(dim=1, keepdim=True)
        P_h = P.sum(dim=0, keepdim=True)

        P   = P.clamp_min(eps)
        P_l = P_l.clamp_min(eps)
        P_h = P_h.clamp_min(eps)

        dc = (P * (torch.log(P) - torch.log(P_l) - torch.log(P_h))).sum()
        return dc
    
    # ---------------- Semantic Smoothing ----------------
    def _compute_semantic_target(self, targets, hand_crafted_text_features):
        """
        Generates soft targets based on CLIP text embedding similarity.
        Resembles LDLVA's label distribution but using Semantic Space as the manifold.
        """
        # 1. Update/Compute Prior Distribution Matrix (Static per epoch theoretically, but we do dynamic for simplicity)
        if self.prior_distribution is None or self.prior_distribution.device != hand_crafted_text_features.device:
            # hand_crafted_text_features: [NumClasses, Dim]
            # Similarity Matrix: [NumClasses, NumClasses]
            sim_matrix = hand_crafted_text_features @ hand_crafted_text_features.t()
            
            # Convert to Probability Distribution (Softmax with Temperature)
            # Sharpeing distribution to avoid over-smoothing
            self.prior_distribution = F.softmax(sim_matrix / self.smoothing_temp, dim=1)

        # 2. Get Prior for Current Batch
        # targets: [BatchSize]
        # batch_prior: [BatchSize, NumClasses]
        batch_prior = self.prior_distribution[targets]
        
        # 3. Mix with One-Hot (Ground Truth)
        # We rely on the concept: Target = (1 - alpha) * OneHot + alpha * Prior
        # This is effectively what Label Smoothing does, but replacing Uniform with Prior.
        
        # One-Hot Target
        batch_size = targets.size(0)
        one_hot = torch.zeros(batch_size, self.num_classes, device=targets.device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Mixed Target
        # label_smoothing parameter acts as 'alpha'
        alpha = self.label_smoothing
        soft_targets = (1.0 - alpha) * one_hot + alpha * batch_prior
        
        return soft_targets

    # ---------------- Forward ----------------
    def forward(
        self,
        logits,                       # learnable logits [B,C]
        targets,
        *,
        epoch: int = None,            # <-- pass epoch để tự set weight
        learnable_text_features=None,
        hand_crafted_text_features=None,
        logits_hand=None,             # handcrafted logits [B,C]
    ):
        if epoch is not None:
            self.set_epoch(int(epoch))

        targets = self._sanitize_targets(targets)
        
        # --- Logit Clamping (Safety) ---
        # Clamp logits to avoid overflow in exp() during softmax
        logits = torch.clamp(logits, min=-30.0, max=30.0)
        if logits_hand is not None:
            logits_hand = torch.clamp(logits_hand, min=-30.0, max=30.0)
        
        # --- Logit Adjustment (Menon et al., 2020) ---
        # Correct formula: logits = logits - tau * log(priors)
        # Reason: log(priors) is negative. Subtracting a negative value adds to the logits of rare classes.
        if self.logit_adjust_tau > 0.0 and self.class_priors is not None:
             # Ensure priors are on the correct device and avoid log(0)
             priors = self.class_priors.to(logits.device)
             logits = logits - self.logit_adjust_tau * torch.log(priors + 1e-8)

        # --- Cross Entropy / Semantic Smoothing ---
        if self.ce_loss is not None:
            # Standard Path (CE or Focal)
            ce = self.ce_loss(logits, targets)
        else:
            # Semantic Smoothing Path
            if hand_crafted_text_features is not None:
                # Ensure hand_crafted_text_features is float for matmul
                if hand_crafted_text_features.dtype != torch.float32:
                    hand_crafted_text_features = hand_crafted_text_features.float()
                
                soft_targets = self._compute_semantic_target(targets, hand_crafted_text_features)
                
                # KLDivLoss expects log-probabilities as input
                log_probs = F.log_softmax(logits, dim=1)
                
                # Apply Class Weights manually if they exist
                if self.class_weights is not None:
                    # Weighting: we can weight the loss per sample based on its true class
                    # Sample weights: [BatchSize]
                    sample_weights = self.class_weights[targets]
                    # KLDiv: sum(soft_targets * (log_target - log_probs)) -> sum(soft_targets * log(soft_targets/probs))
                    # Simplified: - sum(soft_targets * log_probs) (Cross Entropy with Soft Targets)
                    # We implement weighted CE with soft targets manually:
                    per_sample_loss = -torch.sum(soft_targets * log_probs, dim=1)
                    ce = (per_sample_loss * sample_weights).mean()
                else:
                    # Standard CE with soft targets
                    ce = -torch.sum(soft_targets * log_probs, dim=1).mean()
            else:
                # Fallback if no text features provided
                ce = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, weight=self.class_weights)


        mi = self._mi_loss(learnable_text_features, hand_crafted_text_features)
        dc = self._dc_loss(logits, logits_hand)

        total = ce + self.last_w_mi * mi + self.last_w_dc * dc

        return {
            "total": total,
            "ce": ce,
            "mi": mi,
            "dc": dc,
            "w_mi": float(self.last_w_mi),
            "w_dc": float(self.last_w_dc),
        }


def build_criterion(args, mi_estimator=None, num_classes=5):
    return CLIPCAERLoss(args, mi_estimator=mi_estimator, num_classes=num_classes)