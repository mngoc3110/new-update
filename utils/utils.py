# utils/utils.py
import os
import time
import logging
from typing import Optional

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# =========================
# Meter utils
# =========================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path: Optional[str] = None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = "\t".join(entries)

        print(msg)
        if self.log_txt_path is not None:
            with open(self.log_txt_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# =========================
# RecorderMeter (TRAIN only)
# =========================
class RecorderMeter(object):
    """
    Store TRAIN loss / WAR / UAR per epoch.
    (VAL removed – paper-style TRAIN/TEST only)
    """
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.reset(total_epoch)

    def reset(self, total_epoch=None):
        if total_epoch is not None:
            self.total_epoch = total_epoch

        self.epoch = 0
        self.train_loss = []
        self.train_war = []
        self.train_uar = []

    def update(self, epoch, train_loss, train_war, train_uar):
        self.epoch = epoch
        self.train_loss.append(float(train_loss))
        self.train_war.append(float(train_war))
        self.train_uar.append(float(train_uar))


# =========================
# Checkpoint utils
# =========================
def save_checkpoint(state: dict, filename: str):
    """
    Save checkpoint to `filename`.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(path: str, map_location="cpu"):
    """
    Safe checkpoint loading (PyTorch >=2.6 compatible).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


# =========================
# Metrics (WAR / UAR) + CM + report
# =========================
def _unwrap_logits(model_out):
    """
    Model forward may return:
      - Tensor
      - (logits, ...)
      - dict with 'logits' / 'output'
    """
    if isinstance(model_out, torch.Tensor):
        return model_out

    if isinstance(model_out, (list, tuple)) and len(model_out) > 0:
        if isinstance(model_out[0], torch.Tensor):
            return model_out[0]

    if isinstance(model_out, dict):
        for k in ["logits", "output", "pred", "yhat"]:
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]

    raise TypeError(f"Cannot unwrap logits from type={type(model_out)}")


@torch.no_grad()
def computer_uar_war(
    model,
    device=None,
    test_loader=None,
    data_loader=None,
    class_names=None,
    desc="METRICS",
    log_txt_path: Optional[str] = None,
    log_confusion_matrix_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    inference_threshold_binary: float = -1.0, # Renamed parameter
):
    """
    TRAIN / TEST ONLY.
    Priority: test_loader > data_loader
    """

    # ---- choose loader ----
    if test_loader is not None:
        loader = test_loader
        split_name = "TEST"
    elif data_loader is not None:
        loader = data_loader
        split_name = "DATA"
    else:
        raise TypeError("computer_uar_war: need test_loader or data_loader")

    # ---- device ----
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds, all_targets = [], []

    # ---- Open log file for binary probs ----
    bin_prob_log_file = None
    if split_name in ["TEST", "VALID"]:
        bin_prob_log_file = open("binary_probs.log", "a", encoding="utf-8")

    start = time.time()
    for images_face, images_body, target in loader:
        images_face = images_face.to(device, non_blocking=True)
        images_body = images_body.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(images_face, images_body)
        
        # Handle dict output from Hierarchical Model
        if isinstance(out, dict):
            logits = out["logits"]
            logits_binary = out.get("logits_binary", None)
        else:
            logits = _unwrap_logits(out)
            logits_binary = None

        # Hierarchical Inference Logic
        if inference_threshold_binary > 0 and logits_binary is not None:
            probs_binary = torch.softmax(logits_binary, dim=1)
            # Probability of being Non-Neutral (Class 1)
            prob_non_neutral = probs_binary[:, 1]
            
            # ---- Log binary probs ----
            if bin_prob_log_file is not None:
                for i in range(len(prob_non_neutral)):
                    is_neutral_true = 1 if target[i].item() == 0 else 0
                    bin_prob_log_file.write(f"{prob_non_neutral[i].item():.4f},{is_neutral_true}\n")

            # If prob(non-neutral) < threshold -> Force Neutral (0)
            # Otherwise, predict among the emotional classes (1 to 4)
            emotional_preds = torch.argmax(logits[:, 1:], dim=1) + 1
            preds = torch.where(
                prob_non_neutral >= inference_threshold_binary,
                emotional_preds,
                torch.tensor(0, device=device)
            )
        # Fallback to old logic if no binary head but threshold is set (less likely now but safe)
        elif inference_threshold_binary > 0 and logits_binary is None:
             probs = torch.softmax(logits, dim=1)
             non_neutral_probs = probs[:, 1:]
             max_non_neutral_prob, max_non_neutral_idx = torch.max(non_neutral_probs, dim=1)
             adjusted_preds = max_non_neutral_idx + 1
             preds = torch.where(max_non_neutral_prob >= inference_threshold_binary, adjusted_preds, torch.tensor(0, device=device))
        else:
            preds = preds_main
            
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

    # ---- Close log file ----
    if bin_prob_log_file is not None:
        bin_prob_log_file.close()

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # DEBUG: Print unique values to debug "Number of classes mismatch" error
    # print(f"[DEBUG] Unique Targets: {np.unique(all_targets)}")
    # print(f"[DEBUG] Unique Preds: {np.unique(all_preds)}")

    cm = confusion_matrix(all_targets, all_preds)

    war = (cm.diagonal().sum() / max(cm.sum(), 1)) * 100.0
    per_class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    uar = float(np.mean(per_class_recall)) * 100.0

    if class_names is None:
        report = classification_report(all_targets, all_preds, digits=4)
    else:
        report = classification_report(
            all_targets, all_preds, target_names=class_names, digits=4
        )

    elapsed = time.time() - start
    msg = f"[{split_name}][{desc}] time={elapsed:.2f}s | WAR={war:.3f}% | UAR={uar:.3f}%"
    logging.info(msg)

    # ---- PRINT ----
    print("\n" + msg)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    # ---- LOG ----
    if log_txt_path:
        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write("\n" + msg + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            f.write("Classification report:\n")
            f.write(report + "\n")

    return war, uar, cm, report


# =========================
# LR helper
# =========================
def get_lr(optimizer):
    return [pg["lr"] for pg in optimizer.param_groups]

def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Spherical Linear Interpolation (SLERP)
    v0, v1: Tensors to interpolate between
    t: Interpolation factor (scalar or tensor)
    """
    # Normalize vectors
    v0 = v0 / (v0.norm(dim=-1, keepdim=True) + 1e-6)
    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-6)

    # Dot product
    dot = (v0 * v1).sum(dim=-1)

    # If the vectors are too close, use linear interpolation
    if (torch.abs(dot) > DOT_THRESHOLD).any():
        res = (1 - t) * v0 + t * v1
        return res / (res.norm(dim=-1, keepdim=True) + 1e-6)

    # SLERP formula
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    a = torch.sin((1 - t) * theta) / sin_theta
    b = torch.sin(t * theta) / sin_theta
    
    res = a.unsqueeze(-1) * v0 + b.unsqueeze(-1) * v1
    return res