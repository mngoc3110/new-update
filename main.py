# ==================== Imports ====================
import argparse
import datetime
import os
import random
import shutil
import time

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import warnings
from clip import clip
from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from trainer import Trainer
from utils.loss import *
from utils.utils import *
from utils.builders import *

# Ignore specific warnings (for cleaner output)
warnings.filterwarnings("ignore", category=UserWarning)
# Use 'Agg' backend for matplotlib (no GUI required)
matplotlib.use('Agg')

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description='A highly configurable training script for RAER Dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- Experiment and Environment ---
exp_group = parser.add_argument_group('Experiment & Environment', 'Basic settings for the experiment')
exp_group.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help="Execution mode: 'train' for a full training run, 'eval' for evaluation only.")
exp_group.add_argument('--eval-checkpoint', type=str, default='/media/D/zlm/code/CLIP_CAER/outputs_1/test-[07-09]-[22:24]/model_best.pth',
                       help="Path to the model checkpoint for evaluation mode (e.g., outputs/exp_name/model_best.pth).")
exp_group.add_argument('--inference-threshold-binary', type=float, default=-1.0, help='Threshold for Binary Head (Neutral vs Non-Neutral). If Prob(Non-Neutral) < threshold, predict Neutral. Set to -1.0 to disable.')
exp_group.add_argument('--exper-name', type=str, default='test', help='A name for the experiment to create a unique output folder.')
exp_group.add_argument('--dataset', type=str, default='RAER', help='Name of the dataset to use.')
exp_group.add_argument('--gpu', type=str, default='mps', help='ID of the GPU to use, or "mps"/"cpu".')
exp_group.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
exp_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
exp_group.add_argument('--resume', type=str, default=None, help='Path to latest checkpoint (model.pth) to resume training from.')

# --- Data & Path ---
path_group = parser.add_argument_group('Data & Path', 'Paths to datasets and pretrained models')
path_group.add_argument('--root-dir', type=str, default='/media/F/FERDataset/AER-DB', help='Root directory of the dataset.')
path_group.add_argument('--train-annotation', type=str, default='RAER/annotation/train.txt', help='Path to training annotation file, relative to root-dir.')
path_group.add_argument('--val-annotation', type=str, default=None, help='Path to validation annotation file. If None, uses test-annotation for validation.')
path_group.add_argument('--test-annotation', type=str, default='RAER/test.txt', help='Path to testing annotation file, relative to root-dir.')
path_group.add_argument('--clip-path', type=str, default='/media/D/zlm/code/single_four/models/ViT-B-32.pt', help='Path to the pretrained CLIP model.')
path_group.add_argument('--bounding-box-face', type=str, default='/media/F/FERDataset/AER-DB/RAER/bounding_box/face_abs.json')
path_group.add_argument('--bounding-box-body', type=str, default="/media/F/FERDataset/AER-DB/RAER/bounding_box/body_abs.json")
path_group.add_argument('--data-percentage', type=float, default=1.0, help='Percentage of the dataset to use for training and validation (e.g., 0.1 for 10%%).')

# --- Training Control ---
train_group = parser.add_argument_group('Training Control', 'Parameters to control the training process')
train_group.add_argument('--epochs', type=int, default=20, help='Total number of training epochs.')
train_group.add_argument('--batch-size', type=int, default=8, help='Batch size for training and validation.')
train_group.add_argument('--print-freq', type=int, default=10, help='Frequency of printing training logs.')
train_group.add_argument('--use-baseline-config', type=str, default='False', choices=['True', 'False'], help='Use the simple, high-LR baseline configuration (no complex re-balancing).')
train_group.add_argument('--distraction-boost', type=float, default=1.0, help='Multiplier to boost the class weight of Distraction (Class 4).')

# --- Optimizer & Learning Rate ---
optim_group = parser.add_argument_group('Optimizer & LR', 'Hyperparameters for the optimizer and scheduler')
optim_group.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate for main modules.')
optim_group.add_argument('--lr-image-encoder', type=float, default=1e-5, help='Learning rate for the image encoder part.')
optim_group.add_argument('--lr-prompt-learner', type=float, default=1e-3, help='Learning rate for the prompt learner.')
optim_group.add_argument('--lr-adapter', type=float, default=1e-3, help='Learning rate for the adapter modules.')
optim_group.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for the optimizer.')
optim_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
optim_group.add_argument('--milestones', nargs='+', type=int, default=[10, 15], help='Epochs at which to decay the learning rate.')
optim_group.add_argument('--gamma', type=float, default=0.1, help='Factor for learning rate decay.')

# --- Model & Input ---
model_group = parser.add_argument_group('Model & Input', 'Parameters for model architecture and data handling')
model_group.add_argument('--text-type', default='class_descriptor', choices=['class_names', 'class_names_with_context', 'class_descriptor'], help='Type of text prompts to use.')
model_group.add_argument('--temporal-layers', type=int, default=1, help='Number of layers in the temporal modeling part.')
model_group.add_argument('--contexts-number', type=int, default=8, help='Number of context vectors in the prompt learner.')
model_group.add_argument('--class-token-position', type=str, default="end", help='Position of the class token in the prompt.')
model_group.add_argument('--class-specific-contexts', type=str, default='True', choices=['True', 'False'], help='Whether to use class-specific context prompts.')
model_group.add_argument('--load_and_tune_prompt_learner', type=str, default='True', choices=['True', 'False'], help='Whether to load and fine-tune the prompt learner.')
model_group.add_argument('--num-segments', type=int, default=16, help='Number of segments to sample from each video.')
model_group.add_argument('--duration', type=int, default=1, help='Duration of each segment.')
model_group.add_argument('--image-size', type=int, default=224, help='Size to resize input images to.')
model_group.add_argument('--unfreeze-visual-last-layer', type=str, default='False', choices=['True', 'False'], help='Unfreeze the last layer of the visual encoder for fine-tuning.')
model_group.add_argument('--use-multi-scale', type=str, default='False', choices=['True', 'False'], help='Use multi-scale input (Face + Body/Global) for better feature extraction.')
model_group.add_argument('--use-hierarchical-prompt', type=str, default='False', choices=['True', 'False'], help='Enable Lite-HiCroPL 3-level prompt ensemble.')
model_group.add_argument('--use-adapter', type=str, default='False', choices=['True', 'False'], help='Enable Expression-aware Adapter (EAA).')
model_group.add_argument('--use-iec', type=str, default='False', choices=['True', 'False'], help='Enable Instance-enhanced Expression Classifier (IEC).')

# --- Loss & Regularization ---
loss_group = parser.add_argument_group('Loss & Regularization', 'Hyperparameters for loss functions and regularization')
loss_group.add_argument('--use-lsr2-loss', type=str, default='False', choices=['True', 'False'], help='Use LSR2 loss (baseline specific) instead of standard CE/Focal Loss.')
loss_group.add_argument('--lambda-mi', type=float, default=1.0, help='Weight for Mutual Information (MI) loss.')
loss_group.add_argument('--lambda-dc', type=float, default=1.0, help='Weight for Distance Correlation (DC) loss.')
loss_group.add_argument('--lambda-cons', type=float, default=0.0, help='Weight for Consistency Loss (KL Divergence).')
loss_group.add_argument('--lambda-binary', type=float, default=1.0, help='Weight for Hierarchical Binary Loss (Neutral vs Non-Neutral).') # New
loss_group.add_argument('--mi-warmup', type=int, default=0, help='Epochs to warmup MI loss.')
loss_group.add_argument('--mi-ramp', type=int, default=0, help='Epochs to ramp up MI loss.')
loss_group.add_argument('--dc-warmup', type=int, default=0, help='Epochs to warmup DC loss.')
loss_group.add_argument('--dc-ramp', type=int, default=0, help='Epochs to ramp up DC loss.')
loss_group.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor.')
loss_group.add_argument('--semantic-smoothing', type=str, default='True', choices=['True', 'False'], help='Whether to use semantic-guided label smoothing (LDLVA-inspired).')
loss_group.add_argument('--smoothing-temp', type=float, default=0.1, help='Temperature for semantic label distribution (lower = sharper).')
loss_group.add_argument('--use-amp', type=str, default='True', choices=['True', 'False'], help='Enable or disable Automatic Mixed Precision (AMP).')
loss_group.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate gradients before updating weights.')
loss_group.add_argument('--use-focal-loss', type=str, default='True', choices=['True', 'False'], help='Use Focal Loss instead of CE for Stage 1.')
loss_group.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')

# --- Four-Stage Training Control ---
stage_group = parser.add_argument_group('Four-Stage Training', 'Parameters for the multi-stage training strategy')
# Stage 1: Warmup (0 -> stage1_epochs)
stage_group.add_argument('--stage1-epochs', type=int, default=15, help='End epoch of Stage 1 (Warmup).')
stage_group.add_argument('--stage1-label-smoothing', type=float, default=0.05, help='Label smoothing factor for Stage 1.')
stage_group.add_argument('--stage1-smoothing-temp', type=float, default=0.15, help='Smoothing temp for Stage 1.')

# Stage 2: Re-balancing (stage1_epochs -> stage2_epochs)
stage_group.add_argument('--stage2-epochs', type=int, default=40, help='End epoch of Stage 2 (Margin Learning).')
stage_group.add_argument('--stage2-logit-adjust-tau', type=float, default=0.4, help='Tau for Logit Adjustment in Stage 2.')
stage_group.add_argument('--stage2-max-class-weight', type=float, default=2.0, help='Max class weight for Weighted Sampler in Stage 2.')
stage_group.add_argument('--stage2-smoothing-temp', type=float, default=0.15, help='Smoothing temp for Stage 2.')
stage_group.add_argument('--stage2-label-smoothing', type=float, default=0.1, help='Label smoothing factor for Stage 2.')

# Stage 3: Aggressive UAR (stage2_epochs -> stage3_epochs)
stage_group.add_argument('--stage3-epochs', type=int, default=75, help='End epoch of Stage 3 (Aggressive UAR).')
stage_group.add_argument('--stage3-logit-adjust-tau', type=float, default=0.8, help='Tau for Logit Adjustment in Stage 3 (Stronger).')
stage_group.add_argument('--stage3-max-class-weight', type=float, default=2.5, help='Max class weight for Weighted Sampler in Stage 3.')
stage_group.add_argument('--stage3-smoothing-temp', type=float, default=0.18, help='Smoothing temp for Stage 3 (Sharper).')

# Stage 4: Cooldown/Polish (stage3_epochs -> end)
stage_group.add_argument('--stage4-logit-adjust-tau', type=float, default=0.5, help='Tau for Logit Adjustment in Stage 4 (Reduced).')
stage_group.add_argument('--stage4-max-class-weight', type=float, default=2.0, help='Max class weight for Stage 4 (Stabilized).')


stage_group.add_argument('--logit-adjust-tau', type=float, default=0.0, help='Initial Tau for Logit Adjustment (usually 0 for Stage 1).')


# ==================== Helper Functions ====================
def setup_environment(args: argparse.Namespace) -> argparse.Namespace:
    if args.gpu == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("MPS not available, falling back to CPU.")
            device = torch.device("cpu")
    elif args.gpu == 'cpu':
        device = torch.device("cpu")
    elif torch.cuda.is_available() and args.gpu.isdigit():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        print(f"CUDA not available or invalid GPU ID '{args.gpu}', falling back to CPU.")
        device = torch.device("cpu")
    
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    
    print("Environment and random seeds set successfully.")
    return args


def setup_paths_and_logging(args: argparse.Namespace) -> argparse.Namespace:
    now = datetime.datetime.now()
    time_str = now.strftime("-[%m-%d]-[%H:%M]")
    
    args.name = args.exper_name + time_str
        
    args.output_path = os.path.join("outputs", args.name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('************************')
    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f'{k} = {v}')
    print('************************')
    
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    with open(log_txt_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k} = {v}\n')
        f.write('*'*50 + '\n\n')
        
    return args

def calculate_weights(class_counts, max_weight, distraction_boost=1.0):
    if class_counts is None:
        return None
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    # w_j = N / (K * n_j)
    weights = total_samples / (num_classes * (class_counts + 1e-6))
    # Clip
    weights = np.clip(weights, a_min=None, a_max=max_weight)
    
    # Apply Distraction Boost (assuming Distraction is index 4 - Class 5)
    if num_classes > 4 and distraction_boost > 1.0:
        weights[4] *= distraction_boost
        print(f"   >>> Applying Distraction Boost (x{distraction_boost}) -> New Distraction Weight: {weights[4]:.4f}")
        
    return weights.tolist()

# ==================== Training Function ====================
def run_training(args: argparse.Namespace) -> None:
    # Check for Baseline Config Override
    use_baseline = str(getattr(args, 'use_baseline_config', 'False')) == 'True'
    if use_baseline:
        print("\n" + "!"*50)
        print("   >>> BASELINE CONFIG ENABLED: Overriding parameters <<<")
        print("   LR: 0.01 | LR_Image: 1e-5 | No Weighted Sampler | No Logit Adj")
        print("!"*50 + "\n")
        args.lr = 0.01
        args.lr_image_encoder = 1e-5
        args.lr_prompt_learner = 0.001
        args.logit_adjust_tau = 0.0
        args.lambda_mi = 0.0
        args.lambda_dc = 0.0
        args.lambda_cons = 0.0
        args.use_focal_loss = 'False'
        
    # Paths for logging and saving
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_curve_path = os.path.join(args.output_path, 'log.png')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')
    checkpoint_path = os.path.join(args.output_path, 'model.pth')
    best_checkpoint_path = os.path.join(args.output_path, 'model_best.pth')        
    best_uar = 0.0
    best_war = 0.0 # Initialize best_war here to avoid UnboundLocalError
    start_epoch = 0
    recorder = RecorderMeter(args.epochs)

    # Load checkpoint if resuming training
    if args.resume:
        if os.path.isfile(args.resume):
            print(f="=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False) # ADDED weights_only=False
            start_epoch = checkpoint['epoch']
            best_uar = checkpoint['best_acc'] # Assuming best_acc stores UAR
            if 'best_war' in checkpoint:
                best_war = checkpoint['best_war']
            # Load recorder state (optional, if you want to continue plot/history)
            if 'recorder' in checkpoint:
                recorder = checkpoint['recorder']
            print(f="=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f="=> No checkpoint found at '{args.resume}', starting from scratch.")
            
    # --- STAGE 1 SETUP (Safe Base Learning) ---
    print("=> INITIALIZING STAGE 1: Warm-up (Epoch 0 - {})".format(args.stage1_epochs))
    # Standard Setup: No Weighted Sampler, No Weights, No Logit Adj
    args.logit_adjust_tau = 0.0
    args.smoothing_temp = args.stage1_smoothing_temp
    args.label_smoothing = args.stage1_label_smoothing
    
    # Disable MI/DC in Stage 1 initially (or very low)
    args.lambda_mi = 0.0
    args.lambda_dc = 0.0
    
    # Load data first to get class counts
    print("=> Building dataloaders (Stage 1: Random Shuffle)...")
    # Force use_weighted_sampler=False if baseline config is active
    force_random = use_baseline
    train_loader, val_loader, test_loader_final = build_dataloaders(args, use_weighted_sampler=False)
    print("=> Dataloaders built successfully.")

    # Pre-calculate class stats
    class_counts = None
    if args.dataset == 'RAER':
        try:
            train_dataset = train_loader.dataset
            labels = [record.label - 1 for record in train_dataset.video_list]
            class_counts = np.bincount(labels)
            args.class_counts = class_counts.tolist() 
            print(f"   Class Counts: {class_counts}")
            
            # --- STAGE 1 WEIGHTS (None or very mild 1.2) ---
            # We use None to be strictly "Warm-up" as requested
            args.class_weights = None 
            print(f"   [Stage 1] Weights OFF (Standard CE)")
            
        except Exception as e:
            print(f"Warning: Could not calculate class counts. Error: {e}")
            args.class_counts = None
            args.class_weights = None

    # Build model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)

    # Load model state from checkpoint if resuming
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False) # ADDED weights_only=False
        model.load_state_dict(checkpoint['state_dict'])
        print(f="=> Model state loaded from '{args.resume}'")

    print("=> Model built and moved to device successfully.")

    # Loss and optimizer
    # Note: args.class_weights is None for Stage 1
    criterion = build_criterion(args, mi_estimator=model.mi_estimator, num_classes=len(class_names)).to(args.device)
    
    # Store original lambdas to restore later
    original_lambda_mi = 0.5 # Default hardcoded in .sh if not passed, assuming we want to ramp to 0.5
    original_lambda_dc = 0.5

    params_to_optimize = [
        {"params": model.temporal_net.parameters(), "lr": args.lr},
        {"params": model.temporal_net_body.parameters(), "lr": args.lr},
        {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
        {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
        {"params": model.project_fc.parameters(), "lr": args.lr_image_encoder},
        {"params": model.binary_head.parameters(), "lr": args.lr}
    ]

    if model.use_adapter:
        params_to_optimize.append({"params": model.adapter.parameters(), "lr": args.lr_adapter})

    if model.use_iec:
        params_to_optimize.append({"params": [model.slerp_t], "lr": args.lr})

    optimizer = torch.optim.SGD(params_to_optimize, momentum=args.momentum, weight_decay=args.weight_decay)

    # Load optimizer state from checkpoint if resuming
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False) # ADDED weights_only=False
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f="=> Optimizer state loaded from '{args.resume}'")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    
    # Load scheduler state from checkpoint if resuming
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False) # ADDED weights_only=False
        if 'scheduler' in checkpoint: # Scheduler might not always be saved
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f="=> Scheduler state loaded from '{args.resume}'")

    # Trainer
    trainer = Trainer(
        model, criterion, optimizer, scheduler, args.device,
        use_amp=(args.use_amp == 'True'), 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_txt_path=log_txt_path,
        class_names=class_names,
        inference_threshold_binary=args.inference_threshold_binary # Pass the new argument
    )
    
    # Re-evaluate start_epoch and best_uar/best_war if recorder was loaded.
    # Otherwise, these would be initialized to 0.
    if args.resume and os.path.isfile(args.resume) and 'recorder' in checkpoint:
        # Assuming recorder might hold best_acc/best_war that are updated.
        # Otherwise, manually set best_uar/best_war from loaded checkpoint if needed.
        pass # The recorder from checkpoint is now active.

    # --- RESUME LOGIC: Set correct stage parameters based on start_epoch ---
    if not use_baseline and start_epoch > 0:
        print(f="=> Resuming at Epoch {start_epoch}. Setting correct Stage parameters...")
        
        # Check which stage we are in and apply ALL settings for that stage immediately
        
        # STAGE 4 CONFIG
        if start_epoch >= args.stage3_epochs:
            print(f"   [Resume] Applying STAGE 4 Config (Cooldown).")
            args.logit_adjust_tau = args.stage4_logit_adjust_tau
            args.stage2_max_class_weight = args.stage4_max_class_weight
            train_loader, val_loader, test_loader_final = build_dataloaders(args, use_weighted_sampler=True)
            criterion.logit_adjust_tau = args.logit_adjust_tau
            
        # STAGE 3 CONFIG
        elif start_epoch >= args.stage2_epochs:
            print(f"   [Resume] Applying STAGE 3 Config (Aggressive Push).")
            args.logit_adjust_tau = args.stage3_logit_adjust_tau
            args.smoothing_temp = args.stage3_smoothing_temp
            
            args.stage2_max_class_weight = args.stage3_max_class_weight
            train_loader, val_loader, test_loader_final = build_dataloaders(args, use_weighted_sampler=True)
            
            if class_counts is not None:
                args.class_weights = calculate_weights(class_counts, args.stage3_max_class_weight)
                new_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
                criterion.class_weights = new_weights
                if criterion.ce_loss is not None:
                    criterion.ce_loss.weight = new_weights
            
            criterion.logit_adjust_tau = args.logit_adjust_tau
            criterion.smoothing_temp = args.smoothing_temp

        # STAGE 2 CONFIG
        elif start_epoch >= args.stage1_epochs:
            print(f"   [Resume] Applying STAGE 2 Config (Double Re-Weighting).")
            args.logit_adjust_tau = args.stage2_logit_adjust_tau
            args.smoothing_temp = args.stage2_smoothing_temp
            args.label_smoothing = args.stage2_label_smoothing
            
            train_loader, val_loader, test_loader_final = build_dataloaders(args, use_weighted_sampler=True)
            
            if class_counts is not None:
                args.class_weights = calculate_weights(class_counts, args.stage2_max_class_weight)
                new_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
                criterion.class_weights = new_weights
                if criterion.ce_loss is not None:
                    criterion.ce_loss.weight = new_weights
                    criterion.ce_loss.label_smoothing = args.label_smoothing

            criterion.lambda_mi = original_lambda_mi
            criterion.lambda_dc = original_lambda_dc
            criterion.logit_adjust_tau = args.logit_adjust_tau
            criterion.smoothing_temp = args.smoothing_temp
            criterion.label_smoothing = args.label_smoothing
            
            # Ensure Priors
            if criterion.class_priors is None and class_counts is not None:
                counts_t = torch.tensor(class_counts, dtype=torch.float32).to(args.device)
                priors = counts_t / counts_t.sum()
                criterion.register_buffer("class_priors", priors)

    for epoch in range(start_epoch, args.epochs):
        
        # Bypass stage logic if using baseline config
        if not use_baseline:
            # ==========================================
            # 4-STAGE TRANSITION LOGIC (REFINED)
            # ==========================================
            
            # --- Enter STAGE 2: DRW (Deferred Re-Weighting) ---
            if epoch == args.stage1_epochs:
                print("\n" + "="*50)
                print(f"   >>> TRANSITIONING TO STAGE 2: DOUBLE RE-WEIGHTING (Epoch {epoch}) <<<")
                print("="*50)
                
                # 1. Update Params
                args.logit_adjust_tau = args.stage2_logit_adjust_tau
                args.smoothing_temp = args.stage2_smoothing_temp
                args.label_smoothing = args.stage2_label_smoothing
                
                # 2. Switch Sampler (Weighted)
                print(f"   [Stage 2] Switching to WeightedRandomSampler (Max Weight {args.stage2_max_class_weight})...")
                train_loader, val_loader, test_loader_final = build_dataloaders(args, use_weighted_sampler=True)
                
                # 3. Weights: ON (Double Reweighting Strategy) - Force the model to learn!
                if class_counts is not None:
                    args.class_weights = calculate_weights(class_counts, args.stage2_max_class_weight)
                    print(f"   [Stage 2] Class Weights: ON (Double Penalty Active). Weights: {np.round(args.class_weights, 4)}")
                
                # 4. Ramp up MI/DC
                print(f"   [Stage 2] Ramping up MI/DC Loss...")
                criterion.lambda_mi = original_lambda_mi
                criterion.lambda_dc = original_lambda_dc
                
                # 5. Update Criterion
                criterion.logit_adjust_tau = args.logit_adjust_tau
                criterion.smoothing_temp = args.smoothing_temp
                criterion.label_smoothing = args.label_smoothing
                
                # Apply Weights explicitly
                if args.class_weights is not None:
                    new_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
                    criterion.class_weights = new_weights
                    if criterion.ce_loss is not None:
                        criterion.ce_loss.weight = new_weights
                        criterion.ce_loss.label_smoothing = args.label_smoothing
                
                # Ensure Priors for Logit Adj
                if criterion.class_priors is None and class_counts is not None:
                    counts_t = torch.tensor(class_counts, dtype=torch.float32).to(args.device)
                    priors = counts_t / counts_t.sum()
                    criterion.register_buffer("class_priors", priors)
                
                print("   [Stage 2] Transition Complete.\n")

            # --- Enter STAGE 3: Targeted Push ---
            elif epoch == args.stage2_epochs:
                print("\n" + "="*50)
                print(f"   >>> TRANSITIONING TO STAGE 3: AGGRESSIVE DOUBLE PUSH (Epoch {epoch}) <<<")
                print("="*50)
                
                # 1. Update Params (Aggressive)
                args.logit_adjust_tau = args.stage3_logit_adjust_tau
                args.smoothing_temp = args.stage3_smoothing_temp 
                
                # 2. Update Sampler
                print(f"   [Stage 3] Updating WeightedRandomSampler (Max Weight {args.stage3_max_class_weight})...")
                args.stage2_max_class_weight = args.stage3_max_class_weight 
                train_loader, val_loader, test_loader_final = build_dataloaders(args, use_weighted_sampler=True)
                
                # 3. Weights: ON (Stronger)
                if class_counts is not None:
                    args.class_weights = calculate_weights(class_counts, args.stage3_max_class_weight)
                    print(f"   [Stage 3] Class Weights: MAXIMUM PENALTY. Weights: {np.round(args.class_weights, 4)}")
                
                # 4. Update Criterion
                criterion.logit_adjust_tau = args.logit_adjust_tau
                criterion.smoothing_temp = args.smoothing_temp
                
                if args.class_weights is not None:
                    new_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
                    criterion.class_weights = new_weights
                    if criterion.ce_loss is not None:
                        criterion.ce_loss.weight = new_weights

                print("   [Stage 3] Transition Complete.\n")

            # --- Enter STAGE 4: Cooldown & Polish ---
            elif epoch == args.stage3_epochs:
                print("\n" + "="*50)
                print(f"   >>> TRANSITIONING TO STAGE 4: Cooldown & Polish (Epoch {epoch}) <<<")
                print("="*50)
                
                # 1. Update Params (Stable)
                args.logit_adjust_tau = args.stage4_logit_adjust_tau
                
                # 2. Update Sampler (Cap 2.0 - Reduced)
                print(f"   [Stage 4] Reducing WeightedRandomSampler (Max Weight {args.stage4_max_class_weight})...")
                args.stage2_max_class_weight = args.stage4_max_class_weight
                train_loader, val_loader, test_loader_final = build_dataloaders(args, use_weighted_sampler=True)
                
                # 3. Weights: Still OFF
                
                # 4. Update Criterion
                criterion.logit_adjust_tau = args.logit_adjust_tau
                # Temp can stay or reduce slightly, keeping it stable
                
                print("   [Stage 4] Transition Complete. LR should be decaying now.\n")

        inf = f'******************** Epoch: {epoch} ********************'
        start_time = time.time()
        print(inf)
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')

        # Log current learning rates
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        lr_str = ' '.join([f'{lr:.1e}' for lr in current_lrs])
        log_msg = f'Current learning rates: {lr_str}'
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n')
        print(log_msg)

        # Train & Validate
        train_war, train_uar, train_los, _ = trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, val_los, _ = trainer.validate(val_loader, str(epoch))
        scheduler.step()

        # Update best metrics
        if val_uar > best_uar:
            best_uar = val_uar
        if val_war > best_war:
            best_war = val_war
        
        is_best = val_uar >= best_uar # Keep using UAR for is_best criterion or change as needed

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_uar,
            'best_war': best_war, 
            'optimizer': optimizer.state_dict(),
            'recorder': recorder
        }, checkpoint_path)
        
        if is_best:
            shutil.copyfile(checkpoint_path, best_checkpoint_path)

        # Record metrics
        epoch_time = time.time() - start_time
        # RecorderMeter in utils.py takes (epoch, train_loss, train_war, train_uar)
        recorder.update(epoch, train_los, train_war, train_uar)
        # recorder.plot_curve(log_curve_path) # plot_curve method is missing in utils.py

        # Log results
        log_msg = (
            f'Train WAR: {train_war:.2f}% | Train UAR: {train_uar:.2f}%\n'
            f'Valid WAR: {val_war:.2f}% | Valid UAR: {val_uar:.2f}%\n'
            f'Best Valid WAR: {best_war:.2f}% | Best Valid UAR: {best_uar:.2f}%\n'
            f'Epoch time: {epoch_time:.2f}s\n'
        )
        print(log_msg)
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n')

    # Final evaluation with best model
    pre_trained_dict = torch.load(best_checkpoint_path,map_location=args.device)['state_dict']
    model.load_state_dict(pre_trained_dict)
    computer_uar_war(
        val_loader=test_loader_final, # Changed to test_loader_final for final evaluation
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset} (Final Test Set)",
        inference_threshold_binary=args.inference_threshold_binary # Pass the threshold
    )

def run_eval(args: argparse.Namespace) -> None:
    print("=> Starting evaluation mode...")
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')

    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)

    # Load pretrained weightsaaaaa
    ckpt = torch.load(args.eval_checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    # Load data
    _, _, test_loader_final = build_dataloaders(args) # Unpack all three, but only use test_loader_final

    # Run evaluation
    computer_uar_war(
        test_loader=test_loader_final,
        model=model,
        device=args.device,
        class_names=class_names, # Pass class_names
        log_txt_path=log_txt_path,
        log_confusion_matrix_path=log_confusion_matrix_path, # Pass log_confusion_matrix_path
        title=f"Confusion Matrix on {args.dataset} (Final Test Set)",
        inference_threshold_binary=args.inference_threshold_binary # Pass the threshold
    )
    print("=> Evaluation complete.")


# ==================== Entry Point ====================
if __name__ == '__main__':
    args = parser.parse_args()
    args = setup_environment(args)
    args = setup_paths_and_logging(args)
    
    if args.mode == 'eval':
        run_eval(args)
    else:
        run_training(args)
