
import os
import re
import gc
import pdb
import copy
import torch
import pickle
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import complete_box_iou_loss
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from einops import rearrange
from model.multiscale import MultiScaleEncoder


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers -1)
        self.layers = nn.ModuleList(nn.Linear(n,k) for n,k in zip([input_dim] +h, h + [output_dim]))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i<self.num_layers -1 else layer(x)
        return x
    

class MSEnc(nn.Module):
    def __init__(self,  sm_patch_size=144, sm_embd=2048, m_patch_size=256, m_embd=1152, lg_patch_size=576, lg_embd=512, embd=768):
        super().__init__()
        
        self.lg_patch = lg_patch_size
        self.lg_embd = lg_embd

        self.sm_patch = sm_patch_size
        self.sm_embd = sm_embd

        self.m_patch = m_patch_size
        self.m_embd = m_embd
        
        self.embd = embd

        self.ms = MultiScaleEncoder(depth=6,                            
                    dim=embd,
                    mlp_dim=512,
                    cross_attn_heads=8,
                    cross_attn_depth=4,
                    cross_attn_dim_head = 64,
                    dropout = 0.1)

        self.sm_norm = nn.LayerNorm(sm_embd, eps=1e-5)
        self.sm_proj = nn.Linear(sm_embd, embd)

        self.m_norm = nn.LayerNorm(m_embd, eps=1e-5)
        self.m_proj = nn.Linear(m_embd, embd)

        self.lg_norm = nn.LayerNorm(lg_embd, eps=1e-5)
        self.lg_proj = nn.Linear(lg_embd, embd)

        self.norm = nn.LayerNorm(embd, eps=1e-5)

        self.position_enc_sm = nn.Parameter(torch.randn(1, sm_patch_size, sm_embd))
        self.position_enc_m = nn.Parameter(torch.randn(1, m_patch_size, m_embd))
        self.position_enc_lg = nn.Parameter(torch.randn(1, lg_patch_size, lg_embd))
    
    def forward(self, x):
        b, p, d = x.shape
        sm_token = x 
        m_token = x.reshape((b, self.m_patch, self.m_embd)) + self.position_enc_m
        lg_token = x.reshape((b, self.lg_patch, self.lg_embd)) + self.position_enc_lg
        sm_token = sm_token + self.position_enc_sm

        sm_token = self.norm(self.sm_proj(self.sm_norm(sm_token)))
        m_token = self.norm(self.m_proj(self.m_norm(m_token)))
        lg_token = self.norm(self.lg_proj(self.lg_norm(lg_token)))

        encoder_output = self.ms(sm_token, m_token, lg_token)
        return encoder_output



class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, debug=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        self_attn_weights = None
        
        if self.debug: print("(CTEL) ||| --5a: Initial src shape:", src.shape, "Contains NaN:", torch.isnan(src).any().item())
        
        if self.self_attn is not None:
            output, _ = self.self_attn(output, output, output, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            is_nan = torch.isnan(output).any().item()
            if self.debug: print("(CTEL) ||| --5b: After self-attention, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
            
            # if is_nan:
            #     # Check each component, src_mask and src_key_padding_mask, to see if there are NaNs in these...
            #     try:
            #         is_src_mask_nan = torch.isnan(src_mask).any().item()
            #     except:
            #         is_src_mask_nan = False
            #     try:
            #         is_src_key_padding_mask_nan = torch.isnan(src_key_padding_mask).any().item()
            #     except:
            #         is_src_key_padding_mask_nan = False
                
            #     # Make Print statements if true, labeling ||| --5ba and ||| --5bb
            #     if is_src_mask_nan:
            #         print("||| --5ba: src_mask contains NaN values.")
            #     else:
            #         print(f'||| --5ba: Source Mask: {src_mask}')
            #     if is_src_key_padding_mask_nan:
            #         print("||| --5bb: src_key_padding_mask contains NaN values.")
            #     else:
            #         print(f'||| --5bb: Source Key Padding Mask: {src_key_padding_mask}')
        
        if self.norm1 is not None:
            output = self.norm1(output)
            if self.debug: print("(CTEL) ||| --5c: After norm1, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
        
        if self.dropout is not None:
            output = self.dropout(output)
            if self.debug: print("(CTEL) ||| --5d: After dropout, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
        
        residual = output
        
        if self.linear1 is not None:
            output = self.linear1(output)
            if self.debug: print("(CTEL) ||| --5e: After linear1, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
        
        if self.activation is not None:
            output = self.activation(output)
            if self.debug: print("(CTEL) ||| --5f: After activation, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
        
        if self.dropout is not None:
            output = self.dropout(output)
            if self.debug: print("(CTEL) ||| --5g: After second dropout, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
        
        if self.linear2 is not None:
            output = self.linear2(output)
            if self.debug: print("(CTEL) ||| --5h: After linear2, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
        
        if self.norm2 is not None:
            output = self.norm2(output)
            if self.debug: print("(CTEL) ||| --5i: After norm2, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
        
        output = output + residual
        if self.debug: print("(CTEL) ||| --5j: After adding residual, output shape:", output.shape, "Contains NaN:", torch.isnan(output).any().item())
        
        return output



class CustomTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, debug=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
    
    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        # print("---5a: Initial src shape:", src.shape, "Contains NaN:", torch.isnan(src).any().item())

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )
        # print("---5b: src_key_padding_mask processed. Contains NaN:", 
        #       torch.isnan(src_key_padding_mask).any().item() if src_key_padding_mask is not None else "N/A")

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask

        # Check sparsity fast path conditions
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first:
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps):
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        
        # print("---5c: Sparsity fast path check:", why_not_sparsity_fast_path)

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        # print("---5d: After sparsity fast path processing. Output shape:", output.shape, 
        #       "Contains NaN:", torch.isnan(output).any().item())

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        # Process through each encoder layer
        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            nan = torch.isnan(output).any().item()
            layer_type = type(mod)
            if self.debug: print(f"(CTE) ---5e-{i}: After layer {i}, output shape: {output.shape}, Contains NaN: {nan}, Layer Type:{layer_type}")

        if convert_to_nested:
            output = output.to_padded_tensor(0.)
            if self.debug: print("(CTE) ---5f: Converted nested tensor to padded tensor. Output shape:", output.shape, 
                    "Contains NaN:", torch.isnan(output).any().item())

        if self.norm is not None:
            output = self.norm(output)
            if self.debug: print("(CTE) ---5g: After applying normalization. Output shape:", output.shape, 
                    "Contains NaN:", torch.isnan(output).any().item())
        
        return output



class Encoder(nn.Module):
    def __init__(self, hidden_dim=768, nheads=8, num_encoder_layers=6, dropout_rate=0.1, num_patch=12, multiscale='False', detector=False, debug=True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dropout_rate = dropout_rate
        self.num_patch = num_patch*num_patch
        self.multiscale = multiscale
        self.detector = detector
        self.debug = debug
        
        # Create a positional encoding module
        if not self.multiscale:
            self.position_enc = nn.Parameter(torch.randn(1, self.num_patch, hidden_dim))
            # Create a linear layer for embedding the encoder features
            self.linear_emb = nn.Linear(2048, hidden_dim)
            
            # Create a transformer encoder
            encoder_layer = CustomTransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout_rate, batch_first=True)
            self.encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        else:
            self.msenc = MSEnc(sm_patch_size=144, sm_embd=2048, m_patch_size=256, m_embd=1152, lg_patch_size=576, lg_embd=512, embd=768)
            

        

    def forward(self, inputs):
        # Step 1: Backbone feature extraction
        features = self.backbone(inputs)
        if self.debug: print("(E) Step 1 - Backbone output:", features.shape, "Contains NaN:", torch.isnan(features).any().item())

        # Step 2: Rearrange features
        features = rearrange(features, 'b d p1 p2 -> b (p1 p2) d')
        if self.debug: print("(E) Step 2 - Rearranged features:", features.shape, "Contains NaN:", torch.isnan(features).any().item())

        if not self.multiscale:
            # Step 3: Linear embedding
            encoder_embedding = self.linear_emb(features)
            if self.debug: print("(E) Step 3 - Linear embedding output:", encoder_embedding.shape, "Contains NaN:", torch.isnan(encoder_embedding).any().item())

            # Step 4: Add positional encoding
            encoder_embedding += self.position_enc
            if self.debug: print("(E) Step 4 - Added positional encoding:", encoder_embedding.shape, "Contains NaN:", torch.isnan(encoder_embedding).any().item())

            # Step 5: Transformer encoder
            encoder_outputs = self.encoder(encoder_embedding)
            if self.debug: print("(E) Step 5 - Transformer encoder output:", encoder_outputs.shape, "Contains NaN:", torch.isnan(encoder_outputs).any().item())

            # Step 6: Mean pooling
            encoder_outputs = torch.mean(encoder_outputs, dim=1)
            if self.debug: print("(E) Step 6 - Mean pooled output:", encoder_outputs.shape, "Contains NaN:", torch.isnan(encoder_outputs).any().item())
        else:
            # Multiscale encoding branch
            self.msenc = self.msenc.to(features.device)
            encoder_outputs = self.msenc(features)
            if self.debug: print("(E) Step 5 (Multiscale) - Multiscale encoder output:", encoder_outputs.shape, "Contains NaN:", torch.isnan(encoder_outputs).any().item())

            # # Mean pooling for multiscale outputs
            # encoder_outputs = torch.mean(encoder_outputs, dim=1)
            if self.debug: print("(E) Step 6 (Multiscale) - Mean pooled output:", encoder_outputs.shape, "Contains NaN:", torch.isnan(encoder_outputs).any().item())
            
            if not self.detector:
                encoder_outputs = torch.mean(encoder_outputs, dim=1)
                if self.debug: print(encoder_outputs.shape)
            else:
                if self.debug: print(f'(E) From Encoder (encoder outputs): {encoder_outputs}')
                return encoder_outputs

        return encoder_outputs


if __name__=='__main__':
    abc = torch.rand((16, 3, 256, 576))
    detector = Encoder(multiscale=True)
    abc = detector(abc)

    fed = torch.rand((16, 3, 256, 576))
    fed = detector(fed)

    gamma=3
    alpha=0.7

    target = torch.tensor([0,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0])
    print(target.shape, abc.shape)


    distance = nn.CosineSimilarity(dim=1, eps=1e-5)(abc, fed)
    mi, ma = -1, 1
    distance = (distance-mi)/(ma-mi)
    logit = 1-distance+1e-3
    print('distance is', logit)

    x = torch.zeros(abc.shape[0], 2)
    x[...,0] = logit 
    x[...,1] = 1 - logit

    alpha_t = torch.tensor([alpha, 1-alpha])
   
    nll_loss = nn.NLLLoss(weight=alpha_t, reduction='none')

    log_p = F.log_softmax(x, dim=-1)
    ce = nll_loss(log_p.float(), target.long())
    print(ce)
    all_rows = torch.arange(len(x))
    log_pt = log_p[all_rows, target.long()] # gives uniform values if the inputs are the same. 
    # if high disparity between the inputs, the output is zero. 
    pt = log_pt.exp() # extremely negative inputs give a zero
    print(pt)
    focal_term = (1-pt)**gamma
    loss = focal_term*ce 
  
    loss = torch.sum(loss) 

    print('loss=', loss)
    

