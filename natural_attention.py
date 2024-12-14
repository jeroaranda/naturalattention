import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NaturalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.hidden_size = config.n_embd
        self.head_dim = self.hidden_size // self.n_head
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, 
                use_cache=False, output_attentions=False):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle layer past if provided
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        present = (k, v) if use_cache else None
        
        # Compute raw attention energies
        attention_energies = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_energies = attention_energies + attention_mask
        
        # Store raw energies for optimization
        self.last_attention_energies = attention_energies.detach()
        
        # Regular attention computation
        attention_probs = F.softmax(attention_energies, dim=-1)
        
        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            
        context_layer = torch.matmul(attention_probs, v)
        
        # Reshape output
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)
        
        # Project output
        output = self.out_proj(context_layer)
        
        outputs = (output, present)
        if output_attentions:
            outputs += (attention_probs,)
            
        return outputs

class GPT2NaturalAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NaturalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )
        
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attentions tuples have varying length
        outputs = attn_outputs[1:]
        
        hidden_states = residual + attn_output
        
        # Store attention energies in parameters for optimizer
        for p in self.parameters():
            p._attention_energies = self.attn.last_attention_energies
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_output
        
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs

class AttentionInformedOptimizer(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, energy_scale=0.1):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.energy_scale = energy_scale
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get attention energies if available
                if hasattr(p, '_attention_energies'):
                    # Scale gradient based on attention energies
                    energy_factor = torch.tanh(p._attention_energies.abs().mean() * self.energy_scale)
                    p.grad.data *= (1.0 + energy_factor)
        
        # Perform regular Adam update
        return super().step(closure)