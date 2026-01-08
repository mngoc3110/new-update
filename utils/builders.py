# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    # clip.load expects model name (e.g., "ViT-B/16") or a path to a .pt file.
    # If args.clip_path contains a slash, it's treated as a model name.
    # Otherwise, it's treated as a local path to a .pt file.
    if '/' in args.clip_path: # e.g., "ViT-B/16"
        CLIP_model, _ = clip.load(args.clip_path, device='cpu')
    else: # e.g., "models/ViT-B-16.pt" or "path/to/ViT-B-32.pt"
        CLIP_model, _ = clip.load(args.clip_path, device='cpu')
    
    # ✅ FIX: MPS (Apple Silicon) is unstable with FP16 in some layers. Force Float32.
    if args.gpu == 'mps':
        print("   [System] Detected MPS (Apple Silicon). Converting CLIP model to Float32 for stability.")
        CLIP_model = CLIP_model.float()
        # Explicitly update dtype attribute if it exists, as .float() might not update the custom attribute
        # if hasattr(CLIP_model, 'dtype'):
        #     CLIP_model.dtype = torch.float32


    print("\nInput Text Prompts:")
    for text in input_text:
        print(text)

    print("\nInstantiating GenerateModel...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_params_keywords = ["image_encoder", "temporal_net", "prompt_learner", "temporal_net_body", "project_fc"]
    
    # Unfreeze last visual layer if requested
    if getattr(args, 'unfreeze_visual_last_layer', 'False') == 'True':
        print("   -> Unfreezing visual layers (resblocks.10, 11 + ln_post)...")
        # Add keywords specific to the last transformer block and final layer norm of the visual encoder
        # Note: In GenerateModel, clip_model is usually assigned to self.model or self.image_encoder (if wrapped).
        # Based on Generate_Model.py inspection, it seems CLIP components are inside.
        # Let's target the parameter names directly.
        trainable_params_keywords.append("visual.transformer.resblocks.10")
        trainable_params_keywords.append("visual.transformer.resblocks.11")
        trainable_params_keywords.append("visual.ln_post")

    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    根据数据集和文本类型获取 class_names 和 input_text（用于生成 CLIP 模型文本输入）。

    Returns:
        class_names: 类别名称，用于混淆矩阵等
        input_text: 输入文本，用于传入模型
    """
    if args.dataset == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction.']
        class_names_with_context = class_names_with_context_5
        class_descriptor = class_descriptor_5
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text



def build_dataloaders(args: argparse.Namespace, use_weighted_sampler: bool = False, binary_classification: bool = False, emotional_only: bool = False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    # Helper to determine if a path is absolute or relative to root_dir
    def get_full_path(base_path, relative_path):
        if relative_path and relative_path.startswith('/'): # If it's an absolute path
            return relative_path
        elif relative_path: # If it's a relative path, join with base_path
            return os.path.join(base_path, relative_path)
        return None # Return None if relative_path is None or empty


    train_annotation_file_path = get_full_path(args.root_dir, args.train_annotation)
    val_annotation_file_path = get_full_path(args.root_dir, args.val_annotation)
    test_annotation_file_path = get_full_path(args.root_dir, args.test_annotation)
    
    bounding_box_face_path = get_full_path(args.root_dir, args.bounding_box_face)
    bounding_box_body_path = get_full_path(args.root_dir, args.bounding_box_body)

    print("Loading train data...")
    train_data, train_collate_fn = train_data_loader(
        list_file=train_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,dataset_name=args.dataset,
        bounding_box_face=bounding_box_face_path,bounding_box_body=bounding_box_body_path,
        root_dir=args.root_dir, data_percentage=args.data_percentage,
        binary_classification=binary_classification,
        emotional_only=emotional_only
    )
    
    val_data = None
    val_collate_fn = None
    if val_annotation_file_path:
        print("Loading validation data...")
        # Force data_percentage=1.0 for validation
        val_data, val_collate_fn = test_data_loader( 
            list_file=val_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,
            bounding_box_face=bounding_box_face_path,bounding_box_body=bounding_box_body_path,
            root_dir=args.root_dir, data_percentage=1.0,
            binary_classification=binary_classification,
            emotional_only=emotional_only
        )
    
    print("Loading test data (for final evaluation)...")
    # Force data_percentage=1.0 for test
    test_data, test_collate_fn = test_data_loader(
        list_file=test_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=bounding_box_face_path,bounding_box_body=bounding_box_body_path,
        root_dir=args.root_dir, data_percentage=1.0,
        binary_classification=binary_classification,
        emotional_only=emotional_only
    )

    print("Creating DataLoader instances...")
    
    if use_weighted_sampler:
        print("   Using WeightedRandomSampler for training data...")
        # Calculate weights
        try:
            labels = [record.label - 1 for record in train_data.video_list]
            class_counts = np.bincount(labels)
            total_samples = len(labels)
            # Weight for each class: N / n_j
            class_weights_raw = 1.0 / class_counts
            
            # Clip class weights based on args.stage2_max_class_weight (if in Stage 2 context)
            # Default to no clipping if not provided or in Stage 1
            max_weight = getattr(args, 'stage2_max_class_weight', None)
            if max_weight is not None:
                class_weights_raw = np.clip(class_weights_raw, a_min=None, a_max=max_weight)

            sample_weights = [class_weights_raw[l] for l in labels]
            sample_weights = torch.DoubleTensor(sample_weights)
            
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False # Sampler is mutually exclusive with shuffle
        except Exception as e:
            print(f"   Warning: Failed to create WeightedRandomSampler ({e}). Falling back to shuffle=True.")
            sampler = None
            shuffle = True
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=train_collate_fn
    )
    
    # Use test_data for validation if val_data is None
    current_val_data = val_data if val_data is not None else test_data
    current_val_collate_fn = val_collate_fn if val_collate_fn is not None else test_collate_fn

    val_loader = torch.utils.data.DataLoader(
        current_val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=current_val_collate_fn
    )

    # Separate test_loader for final evaluation
    test_loader_final = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=test_collate_fn
    )
    
    return train_loader, val_loader, test_loader_final