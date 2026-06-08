import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ProbCrossAttention(nn.Module):
\
\
\
\
\
       

    def __init__(self, dim, beta: float = 2.35, gate_init: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim * 2)                 
        self.v_proj = nn.Linear(dim, dim * 2)                 
        self.out_proj = nn.Linear(dim, dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.eps = 1e-6
        self.gate = nn.Parameter(torch.tensor(gate_init))                        
        self.beta = beta

    def forward(
            self,
            query,
            context,
            context_mask: torch.Tensor = None,
            query_mask: torch.Tensor = None,
            sample=True,
            num_samples=1,
    ):
        B, Tq, C = query.shape
        _, Tk, _ = context.shape

        Q = self.q_proj(query)              

                               
        K_out = self.k_proj(context)
        K_mu, K_logvar = K_out[..., :C], K_out[..., C:]
        K_mu = self.norm_k(K_mu)
        K_var = F.softplus(K_logvar) + self.eps            

                                 
        V_out = self.v_proj(context)
        V_mu, V_logvar = V_out[..., :C], V_out[..., C:]
        V_mu = self.norm_v(V_mu)
        V_var = F.softplus(V_logvar) + self.eps            

                          
        scale = math.sqrt(C)
        mean_scores = torch.matmul(Q, K_mu.transpose(1, 2)) / scale

        var_penalty = torch.matmul(Q.pow(2), K_var.transpose(1, 2)) / C
        scores = mean_scores - self.beta * torch.sqrt(var_penalty)

        if context_mask is not None:
            context_mask = context_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(
                ~context_mask[:, None, :],
                torch.finfo(scores.dtype).min,
            )

        attn_weights = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        eps = torch.randn_like(V_var)
        V_sample = V_mu + torch.sqrt(V_var) * eps
        out = torch.matmul(attn_weights, V_sample)                        

        gate = torch.sigmoid(self.gate)
        proj_out = self.out_proj(out)
        fused = gate * proj_out + (1 - gate) * query

        if query_mask is not None:
            query_mask = query_mask.to(device=fused.device, dtype=fused.dtype)
            fused = fused * query_mask[:, :, None]

        return fused


class TwoWayTransformerLayer(nn.Module):
    def __init__(self, embed_dim, beta=2.35, gate_init=0.0):
        super().__init__()
        self.cross_attn_img_to_txt = ProbCrossAttention(embed_dim, beta, gate_init)
        self.cross_attn_txt_to_img = ProbCrossAttention(embed_dim, beta, gate_init)

    def forward(self, img_tokens, txt_tokens, txt_mask: torch.Tensor = None):
        img_tokens = self.cross_attn_img_to_txt(
            img_tokens,
            txt_tokens,
            context_mask=txt_mask,
        )
        txt_tokens = self.cross_attn_txt_to_img(
            txt_tokens,
            img_tokens,
            query_mask=txt_mask,
        )
        return img_tokens, txt_tokens


class PVL_Adapter(nn.Module):
    def __init__(self,
                 in_channels_vis: int,
                 in_channels_txt: int,
                 adapter_channels: int,
                 beta: float,
                 gate_init: int):
        super().__init__()

                         
        self.proj_vis_down = nn.Sequential(nn.Linear(in_channels_vis, adapter_channels, bias=False))
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)

                       
        self.proj_vis_up = nn.Linear(adapter_channels, in_channels_vis, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)

                                 
        self.two_way = TwoWayTransformerLayer(adapter_channels, beta, gate_init)

    def forward(self, vis, text, text_mask: torch.Tensor = None):
        v = self.proj_vis_down(vis)
        t = self.proj_txt_down(text)

        v_fused, t_fused = self.two_way(v, t, txt_mask=text_mask)

        vis_out = self.proj_vis_up(v_fused)
        txt_out = self.proj_txt_up(t_fused)

        if text_mask is not None:
            txt_out = txt_out * text_mask.to(
                device=txt_out.device,
                dtype=txt_out.dtype,
            )[:, :, None]

        return vis_out, txt_out


class RMSNorm(nn.Module):
\
\
\
\
\
       

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = x_float * rms
        return x_norm.to(orig_dtype) * self.weight.to(orig_dtype)


class DepthAttentionResidual(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(
            self,
            in_dim: int,
            attn_dim: int,
            num_queries: int,
            gate_init: float = -4.0,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.attn_dim = attn_dim
        self.num_queries = num_queries

                                                              
        self.source_norm = nn.LayerNorm(in_dim)

                                                                    
        self.source_down = nn.Linear(in_dim, attn_dim, bias=False)

                                                                           
                                                           
        self.key_norm = RMSNorm(attn_dim)
        self.value_norm = RMSNorm(attn_dim)

                                                    
                                                                                 
                                                                         
        self.depth_queries = nn.Parameter(torch.zeros(num_queries, attn_dim))

                                                                                
        self.out_norm = nn.LayerNorm(attn_dim)
        self.out_proj = nn.Linear(attn_dim, in_dim, bias=False)

                                                
        self.gates = nn.Parameter(torch.full((num_queries,), float(gate_init)))

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.source_down.weight)

                                                                  
                                                                           
        nn.init.zeros_(self.out_proj.weight)

    def forward(
            self,
            current_tokens: torch.Tensor,
            history_tokens: list,
            query_index: int,
            token_mask: torch.Tensor = None,
    ) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if len(history_tokens) == 0:
            return torch.zeros_like(current_tokens)

        if query_index < 0 or query_index >= self.num_queries:
            raise IndexError(
                f"query_index={query_index} out of range for "
                f"num_queries={self.num_queries}"
            )

                                                                           
                               
        projected_sources = []
        for h in history_tokens:
            h = h.to(dtype=current_tokens.dtype, device=current_tokens.device)
            h = self.source_norm(h)
            h = self.source_down(h)
            projected_sources.append(h)

        sources = torch.stack(projected_sources, dim=0)

                                   
        keys = self.key_norm(sources)
        values = self.value_norm(sources)

                                                     
                    
        query = self.depth_queries[query_index].to(
            dtype=current_tokens.dtype,
            device=current_tokens.device,
        )

                               
                           
                                                                          
        logits = torch.einsum("a,nbta->nbt", query, keys)

                                         
        attn = F.softmax(logits.float(), dim=0).to(dtype=current_tokens.dtype)

                          
        mixed = torch.einsum("nbt,nbta->bta", attn, values)
        mixed = self.out_norm(mixed)

        delta = self.out_proj(mixed)
        delta = self.dropout(delta)

        gate = torch.sigmoid(self.gates[query_index]).to(
            dtype=current_tokens.dtype,
            device=current_tokens.device,
        )

        delta = gate * delta

        if token_mask is not None:
            token_mask = token_mask.to(device=delta.device, dtype=delta.dtype)
            delta = delta * token_mask[:, :, None]

        return delta