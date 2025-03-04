import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-5),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


    
class CrossFormer(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, dim_head, dropout, total_depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.proj_lg = nn.Linear(576, 256)
        self.proj_sm = nn.Linear(144, 256)
        self.pool = nn.MaxPool3d((3,1,1))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim),
                FeedForward(dim, mlp_dim),
                FeedForward(dim, mlp_dim)
            ]))
        self.total_depth = total_depth

    def forward(self, sm_tokens, m_tokens, lg_tokens):

        for attend_sm, attend_m, attend_lg, proj_sm, proj_m, proj_ld in self.layers:
            sm_tokens = attend_sm(self.norm(sm_tokens), self.norm( m_tokens), kv_include_self = True) + sm_tokens
            lg_tokens = attend_lg(self.norm(lg_tokens), self.norm( m_tokens), kv_include_self = True) + lg_tokens # (32, 49, 1024)
            m_tokens = m_tokens + attend_m(self.norm(m_tokens), self.norm(lg_tokens), kv_include_self = True) + attend_m(self.norm(m_tokens), self.norm(sm_tokens), kv_include_self = True) 
        
            sm_tokens = proj_sm(self.norm( sm_tokens)) +  sm_tokens
            m_tokens = proj_m(self.norm( m_tokens)) +  m_tokens
            lg_tokens = proj_ld(self.norm(lg_tokens)) + lg_tokens

        sm_tokens = self.proj_sm(sm_tokens.permute(0,2,1)).permute(0,2,1)
        lg_tokens = self.proj_lg(lg_tokens.permute(0,2,1)).permute(0,2,1)

        tokens = rearrange(torch.stack([sm_tokens, m_tokens, lg_tokens]), 's b p d -> b s p d' )   
        tokens = self.pool(tokens).squeeze(1)    
        return tokens
    

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        mlp_dim,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.cf = CrossFormer(dim = dim, mlp_dim=mlp_dim,depth = cross_attn_depth, 
                            heads = cross_attn_heads, dim_head = cross_attn_dim_head, 
                            dropout = dropout, total_depth=depth)
        
    def forward(self, sm_tokens, m_tokens, lg_tokens):
        tokens = self.cf(sm_tokens, m_tokens, lg_tokens)
        return tokens