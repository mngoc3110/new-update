# trainer.py (REPLACEMENT - ổn định, có CM mỗi epoch, chống NaN)
import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.utils import AverageMeter, ProgressMeter

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device,
                 use_amp=True, gradient_accumulation_steps=1, grad_clip_norm=1.0,
                 class_names=None, log_txt_path=None, inference_threshold_binary=-1.0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.print_freq = 10

        self.accumulation_steps = max(int(gradient_accumulation_steps), 1)
        self.grad_clip_norm = float(grad_clip_norm)

        self.class_names = class_names
        self.log_txt_path = log_txt_path
        # Threshold for Binary Head (Hierarchical Inference)
        # If prob(Non-Neutral) < threshold -> Predict Neutral
        self.inference_threshold_binary = inference_threshold_binary

        # ✅ AMP: chỉ bật khi CUDA (MPS float16 dễ NaN)
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        is_cuda = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        print(f"Trainer initialized: device={self.device.type} | amp={self.use_amp} | accum={self.accumulation_steps} | clip={self.grad_clip_norm} | bin_thresh={self.inference_threshold_binary}")

    def _log(self, msg: str):
        logging.info(msg)
        print(msg)
        if self.log_txt_path:
            with open(self.log_txt_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        if is_train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            prefix = f"Train Epoch: [{epoch_str}]"
        else:
            self.model.eval()
            prefix = f"Valid Epoch: [{epoch_str}]"

        losses = AverageMeter('Loss', ':.4e')
        war_meter = AverageMeter('WAR', ':6.2f')
        progress = ProgressMeter(len(loader), [losses, war_meter], prefix=prefix, log_txt_path=self.log_txt_path)

        all_preds, all_targets = [], []

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for i, batch_data in enumerate(loader):
                if batch_data is None:
                    continue
                images_face, images_body, target = batch_data
                images_face = images_face.to(self.device, non_blocking=True)
                images_body = images_body.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # ✅ autocast chỉ khi CUDA
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    out = self.model(images_face, images_body)
                    logits = out["logits"] if isinstance(out, dict) else out
                    
                    # Get Binary Logits if available
                    logits_binary = out.get("logits_binary") if isinstance(out, dict) else None

                    # epoch int cho warmup/ramp
                    try:
                        epoch_num = int(epoch_str)
                    except ValueError:
                        epoch_num = 0

                    loss_dict = self.criterion(
                        logits, target, epoch=epoch_num,
                        learnable_text_features=(out.get("learnable_text_features") if isinstance(out, dict) else None),
                        hand_crafted_text_features=(out.get("hand_crafted_text_features") if isinstance(out, dict) else None),
                        logits_hand=(out.get("logits_hand") if isinstance(out, dict) else None),
                        logits_binary=logits_binary, # Pass to criterion for Hierarchical Loss
                        is_train=is_train
                    )
                    loss = loss_dict["total"]

                    if self.accumulation_steps > 1:
                        loss = loss / self.accumulation_steps

                # ✅ NaN/Inf guard
                if not torch.isfinite(loss):
                    self._log(f"[WARN] {prefix} iter={i}: loss is NaN/Inf -> skip batch")
                    # Print detailed loss components for debugging
                    if isinstance(loss_dict, dict):
                        debug_info = []
                        for k, v in loss_dict.items():
                            if isinstance(v, torch.Tensor):
                                val = v.item()
                            else:
                                val = v
                            debug_info.append(f"{k}={val}")
                        self._log(f"   [DEBUG] Loss components: {', '.join(debug_info)}")
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if is_train:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # step theo accumulation
                    do_step = ((i + 1) % self.accumulation_steps == 0) or ((i + 1) == len(loader))
                    if do_step:
                        # ✅ grad clip
                        if self.grad_clip_norm > 0:
                            if self.use_amp:
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                        if self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.optimizer.zero_grad(set_to_none=True)

                # --- Inference Logic ---
                if getattr(self.criterion, 'binary_classification_stage', False):
                    # --- STAGE 1: BINARY CLASSIFICATION ---
                    preds = logits_binary.argmax(dim=1)
                    targets_for_metric = (target > 0).long()
                else:
                    # --- STAGE 2 / STANDARD ---
                    preds_main = logits.argmax(dim=1)
                    targets_for_metric = target
                    # Hierarchical Inference (Valid/Test Only)
                    if not is_train and self.inference_threshold_binary > 0 and logits_binary is not None:
                        # Calculate Prob of "Non-Neutral" (Class 1)
                        prob_binary = torch.softmax(logits_binary, dim=1)[:, 1]
                        
                        # If Prob(Non-Neutral) < Threshold -> Force Neutral (0)
                        # Else -> Keep Main Prediction
                        preds = torch.where(
                            prob_binary >= self.inference_threshold_binary, 
                            preds_main, 
                            torch.tensor(0, device=self.device)
                        )
                    else:
                        preds = preds_main

                correct = preds.eq(targets_for_metric).sum().item()
                acc = (correct / target.size(0)) * 100.0

                losses.update(float(loss.item()) * self.accumulation_steps, target.size(0))
                war_meter.update(acc, target.size(0))

                all_preds.append(preds.detach().cpu())
                all_targets.append(target.detach().cpu())

                if i % self.print_freq == 0:
                    progress.display(i)

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        cm = confusion_matrix(all_targets, all_preds)
        war = war_meter.avg
        class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
        uar = float(np.mean(class_recall)) * 100.0

        self._log(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
        self._log("Confusion Matrix:\n" + str(cm))

        # per-class recall
        if self.class_names and len(self.class_names) == cm.shape[0]:
            rec_lines = ["Per-class Recall:"]
            for name, r in zip(self.class_names, class_recall):
                rec_lines.append(f" - {name}: {r*100:.2f}%")
            self._log("\n".join(rec_lines))
        else:
            self._log("Per-class Recall (%): " + str(np.round(class_recall * 100, 2)))

        return war, uar, losses.avg, cm

    def train_epoch(self, train_loader, epoch_num):
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)

    def validate(self, val_loader, epoch_num_str="Final"):
        return self._run_one_epoch(val_loader, str(epoch_num_str), is_train=False)