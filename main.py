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
exp_group.add_argument('--exper-name', type=str, default='Kaggle_ViTB32_LiteHiCroPL_4Stage_SmartPush_100Epochs-Resumed-STAGE3_EXTENDED', help='A name for the experiment to create a unique output folder.')
exp_group.add_argument('--dataset', type=str, default='RAER', help='Name of the dataset to use.')
exp_group.add_argument('--gpu', type=str, default='0', help='ID of the GPU to use, or "mps"/"cpu".')
exp_group.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
exp_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
exp_group.add_argument('--resume', type=str, default='outputs/Kaggle_ViTB32_LiteHiCroPL_4Stage_SmartPush_100Epochs-[01-05]-[01:04]/model.pth', help='Path to latest checkpoint (model.pth) to resume training from.')

# --- Data & Path ---
path_group = parser.add_argument_group('Data & Path', 'Paths to datasets and pretrained models')
path_group.add_argument('--root-dir', type=str, default='/kaggle/input/raer-video-emotion-dataset', help='Root directory of the dataset.')
path_group.add_argument('--train-annotation', type=str, default='/kaggle/input/raer-annot/annotation/train_80.txt', help='Path to training annotation file, relative to root-dir.')
path_group.add_argument('--val-annotation', type=str, default='/kaggle/input/raer-annot/annotation/val_20.txt', help='Path to validation annotation file. If None, uses test-annotation for validation.')
path_group.add_argument('--test-annotation', type=str, default='/kaggle/input/raer-annot/annotation/test.txt', help='Path to testing annotation file, relative to root-dir.')
path_group.add_argument('--clip-path', type=str, default='ViT-B/32', help='Path to the pretrained CLIP model.')
path_group.add_argument('--bounding-box-face', type=str, default='/kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/face.json')
path_group.add_argument('--bounding-box-body', type=str, default="/kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/body.json")
path_group.add_argument('--data-percentage', type=float, default=1.0, help='Percentage of the dataset to use for training and validation (e.g., 0.1 for 10%%).')

# --- Training Control ---
train_group = parser.add_argument_group('Training Control', 'Parameters to control the training process')
train_group.add_argument('--epochs', type=int, default=100, help='Total number of training epochs.')
train_group.add_argument('--batch-size', type=int, default=16, help='Batch size for training and validation.')
train_group.add_argument('--print-freq', type=int, default=50, help='Frequency of printing training logs.')

# --- Optimizer & Learning Rate ---
optim_group = parser.add_argument_group('Optimizer & LR', 'Hyperparameters for the optimizer and scheduler')
optim_group.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for main modules.')
optim_group.add_argument('--lr-image-encoder', type=float, default=1e-06, help='Learning rate for the image encoder part.')
optim_group.add_argument('--lr-prompt-learner', type=float, default=0.0005, help='Learning rate for the prompt learner.')
optim_group.add_argument('--lr-adapter', type=float, default=1e-3, help='Learning rate for the adapter modules.')
optim_group.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay for the optimizer.')
optim_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
optim_group.add_argument('--milestones', nargs='+', type=int, default=[70, 90], help='Epochs at which to decay the learning rate.')
optim_group.add_argument('--gamma', type=float, default=0.1, help='Factor for learning rate decay.')

# --- Model & Input ---
model_group = parser.add_argument_group('Model & Input', 'Parameters for model architecture and data handling')
model_group.add_argument('--text-type', default='class_descriptor', choices=['class_names', 'class_names_with_context', 'class_descriptor'], help='Type of text prompts to use.')
model_group.add_argument('--temporal-layers', type=int, default=2, help='Number of layers in the temporal modeling part.')
model_group.add_argument('--contexts-number', type=int, default=16, help='Number of context vectors in the prompt learner.')
model_group.add_argument('--class-token-position', type=str, default="end", help='Position of the class token in the prompt.')
model_group.add_argument('--class-specific-contexts', type=str, default='True', choices=['True', 'False'], help='Whether to use class-specific context prompts.')
model_group.add_argument('--load_and_tune_prompt_learner', type=str, default='True', choices=['True', 'False'], help='Whether to load and fine-tune the prompt learner.')
model_group.add_argument('--num-segments', type=int, default=16, help='Number of segments to sample from each video.')
model_group.add_argument('--duration', type=int, default=1, help='Duration of each segment.')
model_group.add_argument('--image-size', type=int, default=224, help='Size to resize input images to.')
model_group.add_argument('--unfreeze-visual-last-layer', type=str, default='False', choices=['True', 'False'], help='Unfreeze the last layer of the visual encoder for fine-tuning.')
model_group.add_argument('--use-multi-scale', type=str, default='False', choices=['True', 'False'], help='Use multi-scale input (Face + Body/Global) for better feature extraction.')
model_group.add_argument('--use-hierarchical-prompt', type=str, default='True', choices=['True', 'False'], help='Enable Lite-HiCroPL 3-level prompt ensemble.')
model_group.add_argument('--use-adapter', type=str, default='False', choices=['True', 'False'], help='Enable Expression-aware Adapter (EAA).')
model_group.add_argument('--use-iec', type=str, default='False', choices=['True', 'False'], help='Enable Instance-enhanced Expression Classifier (IEC).')
model_group.add_argument('--binary-classification-stage', type=str, default='False', choices=['True', 'False'], help='Enable Stage 1 Binary Classification (Neutral vs Non-Neutral).')

# --- Loss & Regularization ---
loss_group = parser.add_argument_group('Loss & Regularization', 'Hyperparameters for loss functions and regularization')
loss_group.add_argument('--lambda-mi', type=float, default=0.5, help='Weight for Mutual Information (MI) loss.')
loss_group.add_argument('--lambda-dc', type=float, default=0.5, help='Weight for Distance Correlation (DC) loss.')
loss_group.add_argument('--lambda-cons', type=float, default=0.1, help='Weight for Consistency Loss (KL Divergence).')
loss_group.add_argument('--lambda-binary', type=float, default=0.3, help='Weight for the auxiliary binary head loss.') # Multi-task weight
loss_group.add_argument('--mi-warmup', type=int, default=5, help='Epochs to warmup MI loss.')
loss_group.add_argument('--mi-ramp', type=int, default=0, help='Epochs to ramp up MI loss.')
loss_group.add_argument('--dc-warmup', type=int, default=5, help='Epochs to warmup DC loss.')
loss_group.add_argument('--dc-ramp', type=int, default=0, help='Epochs to ramp up DC loss.')
loss_group.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor.')
loss_group.add_argument('--semantic-smoothing', type=str, default='True', help='Whether to use semantic smoothing.')
loss_group.add_argument('--smoothing-temp', type=float, default=0.1, help='Temperature for semantic smoothing.')
loss_group.add_argument('--use-focal-loss', type=str, default='True', choices=['True', 'False'], help='Use Focal Loss instead of standard Cross Entropy.')
loss_group.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
loss_group.add_argument('--use-amp', type=str, default='True', choices=['True', 'False'], help='Enable or disable Automatic Mixed Precision (AMP).')
loss_group.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate gradients before updating weights.')
loss_group.add_argument('--stage1-epochs', type=int, default=3, help='Number of epochs for stage 1.')
loss_group.add_argument('--stage1-label-smoothing', type=float, default=0.05, help='Label smoothing for stage 1.')
loss_group.add_argument('--stage1-smoothing-temp', type=float, default=0.15, help='Smoothing temperature for stage 1.')
loss_group.add_argument('--stage2-epochs', type=int, default=30, help='Number of epochs for stage 2.')
loss_group.add_argument('--stage2-logit-adjust-tau', type=float, default=0.5, help='Logit adjustment tau for stage 2.')
loss_group.add_argument('--stage2-max-class-weight', type=float, default=2.0, help='Max class weight for stage 2.')
loss_group.add_argument('--stage2-smoothing-temp', type=float, default=0.15, help='Smoothing temperature for stage 2.')
loss_group.add_argument('--stage2-label-smoothing', type=float, default=0.1, help='Label smoothing for stage 2.')
loss_group.add_argument('--stage3-epochs', type=int, default=70, help='Number of epochs for stage 3.')
loss_group.add_argument('--stage3-logit-adjust-tau', type=float, default=0.8, help='Logit adjustment tau for stage 3.')
loss_group.add_argument('--stage3-max-class-weight', type=float, default=5.0, help='Max class weight for stage 3.')
loss_group.add_argument('--stage3-smoothing-temp', type=float, default=0.18, help='Smoothing temperature for stage 3.')
loss_group.add_argument('--stage4-logit-adjust-tau', type=float, default=0.2, help='Logit adjustment tau for stage 4.')
loss_group.add_argument('--stage4-max-class-weight', type=float, default=2.0, help='Max class weight for stage 4.')
loss_group.add_argument('--stage4-focal-gamma', type=float, default=0.5, help='Focal gamma for stage 4.')
loss_group.add_argument('--stage4-use-focal-loss', type=str, default='True', choices=['True', 'False'])
loss_group.add_argument('--stage4-semantic-smoothing', type=str, default='False', choices=['True', 'False'])
loss_group.add_argument('--stage4-smoothing-temp', type=float, default=0.30, help='Smoothing temperature for stage 4.')
loss_group.add_argument('--stage4-neutral-weight', type=float, default=-1.0, help='Weight for neutral class in stage 4. -1 to disable.')
loss_group.add_argument('--inference-neutral-bias', type=float, default=0.0, help='Post-hoc bias for neutral class during inference.')
loss_group.add_argument('--stage3-logit-adjust-tau-neutral', type=float, default=0.1, help='Logit adjustment tau for neutral class in stage 3.')
loss_group.add_argument('--use-lsr2-loss', type=str, default='True', choices=['True', 'False'], help='Use LSR2 loss.')

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

# ==================== Training Function ====================
def run_training(args: argparse.Namespace) -> None:
    # --- Paths and Setup ---
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')
    checkpoint_path = os.path.join(args.output_path, 'model.pth')
    best_checkpoint_path = os.path.join(args.output_path, 'model_best.pth')
    
    best_uar = 0.0
    start_epoch = 0
    recorder = RecorderMeter(args.epochs)
    
    # --- Dataloaders ---
    # For multi-task, we always load the full 5-class dataset
    print("=> Building dataloaders...")
    train_loader, val_loader, test_loader_final = build_dataloaders(args, use_weighted_sampler=False) # Keep sampler simple for now
    print("=> Dataloaders built successfully.")

    # --- Class Weights for Imbalance ---
    args.class_weights = None
    try:
        print("=> Calculating class weights for the loss function...")
        train_dataset = train_loader.dataset
        # Assuming labels are 1-5, mapping to 0-4
        labels = [record.label - 1 for record in train_dataset.video_list]
        class_counts = np.bincount(labels, minlength=5) # Ensure 5 classes
        
    # Effective Number of Samples (ENOS) weighting
        # https://arxiv.org/abs/1901.05555
        beta = 0.999 # A common value for beta, can be tuned
        
        # Calculate effective number for each class
        effective_num = 1.0 - np.power(beta, class_counts)
        
        # Calculate unnormalized weights: (1 - beta) / effective_num
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        
        # Normalize weights to sum to the number of classes (as per common practice)
        # This makes the average weight across classes equal to 1.
        weights = per_cls_weights / np.sum(per_cls_weights) * len(class_counts)
        
        args.class_weights = weights.tolist()
        print(f"   Class Counts: {class_counts}")
        print(f"   Applying Class Weights: {[f'{w:.2f}' for w in args.class_weights]}")
        
    except Exception as e:
        print(f"Warning: Could not calculate class weights. Error: {e}")

    # --- Model ---
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text).to(args.device)

    # --- Optimizer ---
    print("=> Setting up optimizer for multi-task training...")
    # Train all relevant parts: image encoder, adapter, prompts, and both heads.
    params_to_optimize = [
        {"params": model.temporal_net.parameters(), "lr": args.lr},
        {"params": model.temporal_net_body.parameters(), "lr": args.lr},
        {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
        {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
        {"params": model.project_fc.parameters(), "lr": args.lr}, # Main head
        {"params": model.binary_head.parameters(), "lr": args.lr}   # Binary head
    ]
    if model.use_adapter:
        params_to_optimize.append({"params": model.adapter.parameters(), "lr": args.lr_adapter})
    if model.use_iec:
        params_to_optimize.append({"params": [model.slerp_t], "lr": args.lr})
    
    optimizer = torch.optim.SGD(params_to_optimize, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # --- Resume Logic ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f="=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
            start_epoch = checkpoint['epoch']
            best_uar = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'recorder' in checkpoint:
                recorder = checkpoint['recorder']
            print(f="=> Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f="=> No checkpoint found at '{args.resume}', starting from scratch.")
    
    # --- Loss and Trainer ---
    criterion = build_criterion(args, mi_estimator=model.mi_estimator, num_classes=len(class_names)).to(args.device)
    
    trainer = Trainer(
        model, criterion, optimizer, scheduler, args.device,
        use_amp=(args.use_amp == 'True'), 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_txt_path=log_txt_path,
        class_names=class_names
    )

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        inf = f'******************** Epoch: {epoch} ********************'
        start_time = time.time()
        print(inf)
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')

        # Log current learning rates
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        lr_str = ' '.join([f'{lr:.1e}' for lr in current_lrs])
        log_msg = f'Current learning rates: {lr_str}'
        print(log_msg)
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n')

        # Train & Validate
        train_war, train_uar, train_los, _ = trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, val_los, _ = trainer.validate(val_loader, str(epoch))
        scheduler.step()

        # Save best model based on UAR
        is_best = val_uar > best_uar
        best_uar = max(val_uar, best_uar)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_uar,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'recorder': recorder
        }, checkpoint_path)
        
        if is_best:
            shutil.copyfile(checkpoint_path, best_checkpoint_path)

        # Log results
        epoch_time = time.time() - start_time
        log_msg = (
            f'Train WAR: {train_war:.2f}% | Train UAR: {train_uar:.2f}%\n'
            f'Valid WAR: {val_war:.2f}% | Valid UAR: {val_uar:.2f}%\n'
            f'Best Valid UAR: {best_uar:.2f}%\n'
            f'Epoch time: {epoch_time:.2f}s\n'
        )
        print(log_msg)
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n')

    # --- Final Evaluation ---
    print("=> Starting final evaluation with the best model...")
    pre_trained_dict = torch.load(best_checkpoint_path, map_location=args.device)['state_dict']
    model.load_state_dict(pre_trained_dict)
    computer_uar_war(
        val_loader=test_loader_final,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset} (Final Test Set)"
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
