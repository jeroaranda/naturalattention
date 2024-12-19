import torch
import math

class MuonNaturalAttentionOptimizer(torch.optim.Muon):
    """
    Muon Optimizer with Attention-Informed Updates
    
    Incorporates attention energies from transformer layers to modulate 
    parameter updates, building on the natural attention mechanism.
    
    Key features:
    - Leverages attention energies from previous forward passes
    - Dynamically adjusts update scaling based on attention patterns
    - Provides adaptive optimization with structural awareness
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0.01, 
                 energy_scale=0.1, 
                 alpha=0.5):
        """
        Args:
            params (iterable): Iterable of parameters to optimize
            lr (float): Learning rate
            betas (tuple): Coefficients for computing running averages
            eps (float): Term added for numerical stability
            weight_decay (float): Weight decay (L2 penalty)
            energy_scale (float): Scaling factor for attention energies
            alpha (float): Interpolation parameter between standard update and attention-informed update
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha interpolation value: {alpha}")
        
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay,
            energy_scale=energy_scale,
            alpha=alpha
        )
        super().__init__(params, defaults)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('energy_scale', 0.1)
            group.setdefault('alpha', 0.5)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Modifies the standard Muon update by incorporating attention energies.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            alpha = group['alpha']
            energy_scale = group['energy_scale']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Store the original gradient
                grad = p.grad.data
                
                # Compute state if not already present
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Standard Muon update
                denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (state['exp_avg'] / bias_correction1) / denom
                
                # Attention energy integration
                if hasattr(p, '_attention_energies'):
                    # Compute attention energy factor
                    attention_energies = p._attention_energies
                    energy_factor = torch.tanh(
                        attention_energies.abs().mean() * energy_scale
                    )
                    
                    # Modulate update based on attention energies
                    attention_update = update * (1.0 + energy_factor)
                    
                    # Interpolate between standard and attention-informed update
                    update = alpha * update + (1 - alpha) * attention_update
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Update parameters
                p.data.add_(-group['lr'] * update)
        
        return loss
