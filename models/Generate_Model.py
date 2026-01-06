from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.mi_estimator import MIEstimator
from clip import clip
import torch
import torch.nn.functional as F
from models.Text import get_hierarchical_prompts # Import helper

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        # x_q: [Batch, Length_Q, Dim] (Face)
        # x_kv: [Batch, Length_KV, Dim] (Body/Context)
        B, N_q, C = x_q.shape
        B, N_kv, C = x_kv.shape

        q = self.q_proj(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.use_hierarchical_prompt = str(getattr(args, 'use_hierarchical_prompt', 'False')) == 'True'

        # --- Prompts ---
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.hand_crafted_prompts = input_text
        
        # Tokenize on CPU first (since clip_model is on CPU during init)
        tokenized_prompts = clip.tokenize(self.hand_crafted_prompts)
        self.register_buffer("tokenized_hand_crafted_prompts", tokenized_prompts)

        # --- Hierarchical Prompts (Lite-HiCroPL) ---
        if self.use_hierarchical_prompt:
            print("   [Model] Initializing Lite-HiCroPL (3-Level Ensemble)...")
            hier_prompts = get_hierarchical_prompts()
            self.prompts_l1 = hier_prompts['level1']
            self.prompts_l2 = hier_prompts['level2']
            self.prompts_l3 = hier_prompts['level3']
            
            self.tokenized_l1 = clip.tokenize(self.prompts_l1)
            self.tokenized_l2 = clip.tokenize(self.prompts_l2)
            self.tokenized_l3 = clip.tokenize(self.prompts_l3)
            
            self.register_buffer("tokenized_prompts_l1", self.tokenized_l1)
            self.register_buffer("tokenized_prompts_l2", self.tokenized_l2)
            self.register_buffer("tokenized_prompts_l3", self.tokenized_l3)
            
            # Pre-compute fixed embeddings for these levels (since they are hand-crafted/frozen for now)
            # Or if you want to make them learnable, you'd need multiple PromptLearners.
            # For "Lite" version, we keep them fixed or use the main learnable prompt for L3 and fixed for L1/L2.
            # To keep it simple and consistent with "Hand-crafted branch": We treat these as auxiliary fixed prompts 
            # that ensemble with the main learnable branch.
            
            # Actually, to maximize impact, let's pre-compute their embeddings (frozen) 
            # and ensemble them into the 'hand_crafted' branch OR add them to the main logits.
            # Let's ensemble them into the main decision process.
            with torch.no_grad():
                self.embed_l1 = clip_model.token_embedding(self.tokenized_l1).type(clip_model.dtype)
                self.embed_l2 = clip_model.token_embedding(self.tokenized_l2).type(clip_model.dtype)
                self.embed_l3 = clip_model.token_embedding(self.tokenized_l3).type(clip_model.dtype)
            
            self.register_buffer("fixed_embed_l1", self.embed_l1)
            self.register_buffer("fixed_embed_l2", self.embed_l2)
            self.register_buffer("fixed_embed_l3", self.embed_l3)


        # --- Encoders ---
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual

        # --- Hand-crafted token embedding (frozen) ---
        with torch.no_grad():
            embedding = clip_model.token_embedding(
                self.tokenized_hand_crafted_prompts
            ).type(clip_model.dtype)
        self.register_buffer("hand_crafted_embedding", embedding)

        # Fuse face+body -> instead of simple concat, we use Cross Attention
        # Face query Body. 
        self.cross_attn = CrossAttention(dim=512, num_heads=8)
        self.norm_face = nn.LayerNorm(512)
        self.norm_body = nn.LayerNorm(512)
        
        # Final refinement instead of project_fc
        self.project_fc = nn.Linear(512, 512)
        self.dtype = clip_model.dtype 

        # --- Temporal Models ---
        self.temporal_net = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=args.temporal_layers,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
        )

        self.temporal_net_body = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=args.temporal_layers,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
        )

        # --- MI Estimator ---
        self.mi_estimator = MIEstimator(
            feature_dim=clip_model.text_projection.shape[1]
        )

        # ✅ CLIP-style logit scaling (learnable)
        self.logit_scale = clip_model.logit_scale

        # Optional: fixed temperature override (if args.tau is set)
        self.tau = float(getattr(args, "tau", 0.0))  

        # ✅ Check if we need the Hand-crafted branch (View 2)
        self.use_handcrafted_branch = (
            getattr(args, "lambda_mi", 0.0) > 0.0 or
            getattr(args, "lambda_dc", 0.0) > 0.0
        )

        # ✅ Hierarchical Learning: Binary Head (Neutral vs. Others)
        # Input: 512 (video_features), Output: 2
        self.binary_head = nn.Linear(512, 2)

    def forward(self, image_face, image_body, return_dict: bool = True):
        ################# Visual Part #################
        # Face
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        face_feat = self.image_encoder(image_face.type(self.dtype))
        face_feat = face_feat.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(face_feat) # [B, 512]

        # Body
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        body_feat = self.image_encoder(image_body.type(self.dtype))
        body_feat = body_feat.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(body_feat) # [B, 512]

        # Fuse with Cross Attention
        # Query: Face, Key/Value: Body
        # Reshape to [B, 1, 512] for attention
        q = self.norm_face(video_face_features).unsqueeze(1)
        kv = self.norm_body(video_body_features).unsqueeze(1)
        
        # Attention
        fused_feat = self.cross_attn(q, kv).squeeze(1) # [B, 512]
        
        # Residual connection + Projection
        # We add fused features to original face features to keep identity strong
        video_features = video_face_features + fused_feat
        video_features = self.project_fc(video_features)
        
        # L2 Norm
        video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# Binary Head Output #################
        logits_binary = self.binary_head(video_features) # [B, 2]

        ################# Text Part (Dual View) #################
        # 1) Learnable prompts (ALWAYS needed for classification)
        learnable_prompts = self.prompt_learner()
        tokenized_learnable_prompts = self.prompt_learner.tokenized_prompts
        learnable_text_features = self.text_encoder(
            learnable_prompts, tokenized_learnable_prompts
        )
        learnable_text_features = learnable_text_features / (
            learnable_text_features.norm(dim=-1, keepdim=True) + 1e-6
        )
        if learnable_text_features.dtype != video_features.dtype:
            learnable_text_features = learnable_text_features.to(video_features.dtype)

        # 2) Hand-crafted prompts (ONLY needed if MI or DC > 0)
        hand_crafted_text_features = None
        logits_hand = None
        probs_hand = None

        if self.use_handcrafted_branch:
            hand_crafted_text_features = self.text_encoder(
                self.hand_crafted_embedding, self.tokenized_hand_crafted_prompts
            )
            hand_crafted_text_features = hand_crafted_text_features / (
                hand_crafted_text_features.norm(dim=-1, keepdim=True) + 1e-6
            )
            if hand_crafted_text_features.dtype != video_features.dtype:
                hand_crafted_text_features = hand_crafted_text_features.to(video_features.dtype)

        ################# ✅ CLIP-style Logits #################
        scale = self.logit_scale.exp().clamp(1e-3, 100.0)

        if self.tau and self.tau > 0:
            scale = torch.tensor(1.0 / self.tau, device=video_features.device, dtype=video_features.dtype)

        # Main Logits (Learnable)
        logits_learnable = scale * (video_features @ learnable_text_features.t())
        
        # --- Lite-HiCroPL Ensemble ---
        if self.use_hierarchical_prompt:
            # Compute features for 3 levels (Fixed prompts for now to keep compute low)
            # Level 1: Visual
            feat_l1 = self.text_encoder(self.fixed_embed_l1, self.tokenized_prompts_l1)
            feat_l1 = feat_l1 / (feat_l1.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Level 2: Action
            feat_l2 = self.text_encoder(self.fixed_embed_l2, self.tokenized_prompts_l2)
            feat_l2 = feat_l2 / (feat_l2.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Level 3: Abstract
            feat_l3 = self.text_encoder(self.fixed_embed_l3, self.tokenized_prompts_l3)
            feat_l3 = feat_l3 / (feat_l3.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Ensure dtype
            if feat_l1.dtype != video_features.dtype:
                feat_l1 = feat_l1.to(video_features.dtype)
                feat_l2 = feat_l2.to(video_features.dtype)
                feat_l3 = feat_l3.to(video_features.dtype)
            
            # Compute Logits
            logits_l1 = scale * (video_features @ feat_l1.t())
            logits_l2 = scale * (video_features @ feat_l2.t())
            logits_l3 = scale * (video_features @ feat_l3.t())
            
            # Ensemble: Weighted sum
            # We can give slightly more weight to the learnable part, or treat all equally.
            # Strategy: Learnable (Main) + L1 (Visual) + L2 (Action) + L3 (Abstract)
            # Let's average them to keep scale consistent
            
            # Ensemble with the main learnable logits
            # logits_learnable = 0.4 * logits_learnable + 0.2 * logits_l1 + 0.2 * logits_l2 + 0.2 * logits_l3
            # Or simpler:
            logits_learnable = (logits_learnable + logits_l1 + logits_l2 + logits_l3) / 4.0

        probs_learnable = F.softmax(logits_learnable, dim=-1)

        if self.use_handcrafted_branch and hand_crafted_text_features is not None:
            logits_hand = scale * (video_features @ hand_crafted_text_features.t())
            probs_hand = F.softmax(logits_hand, dim=-1)

        if not return_dict:
            return logits_learnable, learnable_text_features, hand_crafted_text_features

        return {
            "logits": logits_learnable,                  
            "logits_learnable": logits_learnable,
            "logits_hand": logits_hand,
            "logits_binary": logits_binary, # ✅ Added binary logits
            "probs_learnable": probs_learnable,
            "probs_hand": probs_hand,
            "video_features": video_features,
            "learnable_text_features": learnable_text_features,
            "hand_crafted_text_features": hand_crafted_text_features,
        }
