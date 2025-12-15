import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import random
from rotary_embedding_torch import RotaryEmbedding


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x):
        freq = torch.einsum('b,d->bd', x, self.weights) * 2 * math.pi
        return torch.cat([freq.sin(), freq.cos()], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, rotary_embedding=None):
        super(MultiHeadAttention, self).__init__()
        assert (dim % num_heads == 0)
        self.model_dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_o = nn.Linear(dim, dim)
        self.rotary_emb = rotary_embedding

    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, _ = q.size()
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        score = (q @ k.transpose(-2, -1)) * 1.0 / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e4)
        score = F.softmax(score, dim=-1)
        out = score @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dim)
        return self.w_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, drop_prob=0.0, rope_dim=64):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=False,
            rotary_embedding=RotaryEmbedding(dim=rope_dim)
        )
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask, gammas=(0.0, 0.0), betas=(0.0, 0.0)):
        res = x
        x = self.norm1(x)
        x = (gammas[0] * x) + betas[0]
        x = self.attention(q=x, k=x, v=x, mask=mask)
        x = res + self.dropout1(x)

        res = x
        x = self.norm2(x)
        x = (gammas[1] * x) + betas[1]
        x = self.ffn(x)
        x = res + self.dropout2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, target_dim, model_dim, num_layers=8, learned_sinusoidal_dim=128, dropout_prob=0.0,
                 layerdrop_prob=0.0):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layerdrop_prob = layerdrop_prob
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
            nn.Linear(learned_sinusoidal_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, num_layers * 4 * model_dim),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob)
        )
        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(
                dim=model_dim,
                hidden_dim=4 * model_dim,
                num_heads=8,
                drop_prob=dropout_prob
            )
            for _ in range(num_layers))
        self.out = nn.Linear(model_dim, target_dim)

    def self_attention_mask(self, length_mask):
        return torch.logical_and(length_mask.unsqueeze(1).unsqueeze(1), length_mask.unsqueeze(1).unsqueeze(-1))

    def forward(self, x, t, length_mask=None):
        time_emb = self.time_mlp(t)
        x = self.project(x)
        attention_mask = None if length_mask is None else self.self_attention_mask(length_mask)
        scaling_weights = time_emb.view(-1, self.num_layers * 4, self.model_dim).split(1, dim=1)
        for i, layer in enumerate(self.encoder_layers):
            if self.training and random.uniform(0, 1) < self.layerdrop_prob:
                continue
            gammas = scaling_weights[4 * i], scaling_weights[4 * i + 1]
            betas = scaling_weights[4 * i + 2], scaling_weights[4 * i + 3]
            x = layer(x, attention_mask, gammas=gammas, betas=betas)

        return self.out(x), x


class Diffusion:
    def __init__(self, estimator: nn.Module, interpolate=None, self_conditioning=False, normalize=False):
        super(Diffusion).__init__()
        self.estimator = estimator
        self.interpolate = interpolate
        self.self_conditioning = self_conditioning
        self.normalize = normalize

    def gamma(self, t, ns=0.0002, ds=0.00025):
        return torch.cos(((t + ns) / (1 + ds)) * math.pi / 2) ** 2

    def forward_diffusion(self, x_0, t):
        time = t.unsqueeze(1).unsqueeze(1)
        mean_weight = torch.sqrt(self.gamma(time))
        std = torch.sqrt(1 - self.gamma(time))
        z = torch.randn_like(x_0)
        x_t = (mean_weight * x_0) + (z * std)
        return x_t, z, std

    def reverse_diffusion(self, x_T, steps):
        """Unconditional reverse diffusion sampling."""
        x_t = x_T
        t_now = torch.ones(x_T.shape[0], dtype=x_t.dtype, device=x_t.device, requires_grad=False)
        
        for step in range(steps):
            if self.normalize:
                x_t = x_t / x_t.std(dim=-1, keepdim=True)
            
            x_estimation, latent = self.estimator(x_t, t_now)
            
            if self.interpolate is not None:
                x_estimation = self.interpolate(latent)
            
            t_next = torch.clamp(t_now - 1 / steps, 0.0, 1.0)
            x_t = self.diff_lm_step(x_estimation, t_next)
            t_now = t_next
        
        t_final = torch.zeros(x_T.shape[0], device=x_T.device)
        _, latent = self.estimator(x_t, t_final)
        return x_t, latent

    def diff_lm_step(self, x_0_estimation, t_next):
        gamma_next = self.gamma(t_next).unsqueeze(1).unsqueeze(1)
        eps = torch.randn_like(x_0_estimation)
        return torch.sqrt(gamma_next) * x_0_estimation + torch.sqrt(1 - gamma_next) * eps

    def loss_t(self, x, t, len_mask, cond_mask):
        """Diffusion loss computation."""
        x_target = x.detach()
        x_t, z, std = self.forward_diffusion(x, t)

        if self.normalize:
            x_t = x_t / x_t.std(dim=-1, keepdim=True)

        x_estimation, latent = self.estimator(x_t, t, len_mask)
        
        return ((x_estimation - x_target) ** 2.0).mean(-1), x_estimation, latent

    def compute_loss(self, x_0, len_mask, cond_mask, offset=1e-5):
        t = torch.rand(x_0.shape[0], dtype=x_0.dtype, device=x_0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        loss, x_0_estimation, latent = self.loss_t(x_0, t, len_mask, cond_mask)
        return loss, x_0_estimation, latent


class DiffusionLM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, model_dim, num_layers, dropout_prob, layerdrop_prob, crop_length,
                 embedding_grad_scale=0.5, interpolate_temperature=0.8, label_smoothing=0.1):
        super(DiffusionLM, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.embedding_grad_scale = embedding_grad_scale
        self.interpolate_temperature = interpolate_temperature
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim
        )
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.estimator = TransformerModel(
            input_dim=self.embedding_dim,
            target_dim=self.embedding_dim,
            model_dim=self.model_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            layerdrop_prob=layerdrop_prob
        )
        self.diffusion = Diffusion(
            estimator=self.estimator,
            interpolate=self.interpolate
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        self.lm_head = nn.Linear(self.model_dim, self.num_embeddings)
        self.loss_ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def get_embeddings(self, ids):
        e = self.embedding(ids)
        e = self.norm(e)
        return e

    def get_logits(self, x):
        x = self.dropout(x)
        x = self.lm_head(x)
        return x

    def interpolate(self, x):
        logits = self.get_logits(x) / self.interpolate_temperature
        weights = logits.softmax(dim=-1)
        e = self.embedding.weight
        e = self.norm(e)
        interpolated = torch.einsum('nle,ed->nld', weights, e)
        return interpolated

    def dist_embedding(self, x):
        e = self.embedding.weight
        e = self.norm(e)
        return torch.cdist(x, e)

    def cosine_similarity(self, x):
        e = self.embedding.weight
        e = F.normalize(e, dim=-1)
        x = F.normalize(x, dim=-1)
        cossim = torch.einsum('nld,ed->nle', x, e)
        return cossim

    def compute_loss(self, ids, lengths, conditional_mask=None):
        """Compute diffusion loss."""
        x = self.get_embeddings(ids)
        x = self.embedding_grad_scale * x + (1.0 - self.embedding_grad_scale) * x.detach()
        length_mask = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        num_elems = length_mask.sum()
        loss_diff, x_estimation, latent = self.diffusion.compute_loss(x, length_mask, conditional_mask)
        loss_diff = loss_diff * length_mask
        loss_diff = loss_diff.sum() / num_elems
        
        logits = self.get_logits(latent)
        ids = ids.masked_fill(torch.logical_not(length_mask), -100)
        loss_reconstruction = self.loss_ce(logits.transpose(2, 1), ids)
        accuracy = (logits.argmax(dim=-1) == ids).float().sum() / num_elems
        loss_reconstruction = loss_reconstruction.sum() / num_elems
        loss = loss_diff + loss_reconstruction

        return loss, loss_diff, loss_reconstruction, accuracy

    def forward(self, z, num_steps=500):
        """Forward inference (unconditional generation)."""
        x, latent = self.diffusion.reverse_diffusion(z, num_steps)
        return self.get_logits(latent).argmax(dim=-1)
    
    def inpaint(self, partial_ids, mask, num_steps=500):
        """Sequence inpainting."""
        device = partial_ids.device
        batch_size, seq_len = partial_ids.shape
        
        known_embeddings = self.get_embeddings(partial_ids)
        x_t = torch.randn(batch_size, seq_len, self.embedding_dim, device=device)
        x_t = torch.where(mask.unsqueeze(-1), x_t, known_embeddings)
        
        t_now = torch.ones(batch_size, dtype=x_t.dtype, device=device, requires_grad=False)
        
        for step in range(num_steps):
            x_estimation, latent = self.estimator(x_t, t_now)
            x_estimation = self.interpolate(latent)
            
            t_next = torch.clamp(t_now - 1 / num_steps, 0.0, 1.0)
            
            gamma_next = self.diffusion.gamma(t_next).unsqueeze(1).unsqueeze(1)
            eps = torch.randn_like(x_estimation)
            x_next = torch.sqrt(gamma_next) * x_estimation + torch.sqrt(1 - gamma_next) * eps
            
            gamma_t = self.diffusion.gamma(t_next).unsqueeze(1).unsqueeze(1)
            known_noisy = torch.sqrt(gamma_t) * known_embeddings + torch.sqrt(1 - gamma_t) * torch.randn_like(known_embeddings)
            x_t = torch.where(mask.unsqueeze(-1), x_next, known_noisy)
            
            t_now = t_next
        
        t_final = torch.zeros(batch_size, device=device)
        _, latent = self.estimator(x_t, t_final)
        
        logits = self.get_logits(latent)
        generated_ids = logits.argmax(dim=-1)
        
        result = torch.where(mask, generated_ids, partial_ids)
        
        return result
