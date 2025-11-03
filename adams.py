"""
AdamS: Adam com Freio Geométrico SIGMA
=======================================

Fusão do Adam com SIGMA onde o score σ atua como MODULADOR FINAL
do passo do Adam, não como cálculo paralelo.

Diferença para AdaSIGMA:
- AdaSIGMA: calcula σ e m_t em paralelo, depois combina
- AdamS: usa passo completo do Adam, depois modula por σ

Regra:
    step_adam = (η / √v̂_t) · m̂_t
    step_adamS = step_adam · σ_t  ← σ como freio/acelerador
"""

import torch
from torch.optim.optimizer import Optimizer

class AdamS(Optimizer):
    """
    AdamS: Adam com freio geométrico baseado no score SIGMA.
    
    O score σ atua como modulador:
    - σ < 1: freia o Adam (região não-convexa instável)
    - σ = 1: comportamento idêntico ao Adam
    - σ > 1: acelera o Adam (região convexa favorável)
    
    Args:
        params: Parâmetros a otimizar
        lr: Learning rate (default: 1e-3)
        betas: Coeficientes Adam (β₁, β₂) (default: (0.9, 0.999))
        eps: Estabilidade numérica (default: 1e-8)
        beta_s: Momentum do score σ (default: 0.9)
        alpha_min: Limite inferior de σ (default: 0.1)
        alpha_max: Limite superior de σ (default: 2.0)
        warmup_steps: Épocas antes de ativar σ (default: 10)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 beta_s=0.9, alpha_min=0.5, alpha_max=1.5, warmup_steps=10):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= beta_s < 1.0:
            raise ValueError(f"Invalid beta_s: {beta_s}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, beta_s=beta_s,
                        alpha_min=alpha_min, alpha_max=alpha_max,
                        warmup_steps=warmup_steps)
        
        super(AdamS, self).__init__(params, defaults)
        
        # Estado global
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, closure=None, loss_item=None):
        """
        Executa passo de otimização.
        
        Args:
            closure: Função que reavalia modelo (opcional)
            loss_item: Valor da loss (obrigatório)
        """
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if loss_item is None:
            raise ValueError("AdamS.step() requer 'loss_item'")
            
        loss_t = loss_item
        self.state['global_step'] += 1
        step = self.state['global_step']
        loss_prev = self.state['global_loss_prev']
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            beta_s = group['beta_s']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            warmup_steps = group['warmup_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]

                # Inicialização
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)      # m_t (Adam)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)   # v_t (Adam)
                    state['param_prev'] = p.data.clone()             # θ_{t-1}
                    state['score_momentum'] = torch.ones_like(p.data) # m_s (SIGMA)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                param_prev = state['param_prev']
                score_mom = state['score_momentum']
                
                state['step'] += 1
                
                # ============================================================
                # PARTE 1: Calcular Momentos do Adam (PADRÃO)
                # ============================================================
                
                # m_t = β₁·m_{t-1} + (1-β₁)·g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # v_t = β₂·v_{t-1} + (1-β₂)·g_t²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # m̂_t = m_t / (1 - β₁^t)
                exp_avg_corrected = exp_avg / bias_correction1
                
                # v̂_t = v_t / (1 - β₂^t)
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Denominador: √v̂_t + ε
                denom = exp_avg_sq_corrected.sqrt().add_(eps)
                
                # ============================================================
                # PARTE 2: Calcular Score Geométrico σ (SIGMA)
                # ============================================================
                
                theta_t = p.data  # ← Mover para antes do if
                
                if step <= warmup_steps:
                    # Durante warmup, σ = 1 (comportamento = Adam puro)
                    sigma_raw = torch.ones_like(p.data)
                else:
                    # Pontos D (interpolação harmônica)
                    denom_xy = param_prev + theta_t + eps
                    D1 = (param_prev * theta_t) / denom_xy
                    D2 = (theta_t * loss_prev + param_prev * loss_t) / denom_xy
                    
                    # Aproximação f(2D₁) via Taylor
                    f_proxy = loss_t + grad * (2 * D1 - theta_t)
                    
                    # Score: σ = D₂ / f_proxy
                    sigma_raw = D2 / (f_proxy.abs() + eps)
                    
                    # Remover NaN/Inf
                    sigma_raw = torch.where(
                        torch.isfinite(sigma_raw),
                        sigma_raw,
                        torch.ones_like(sigma_raw)
                    )
                
                # Suavizar score com momentum
                score_mom.mul_(beta_s).add_(sigma_raw, alpha=1 - beta_s)
                
                # Clipping
                sigma_final = score_mom.clamp(alpha_min, alpha_max)
                
                # ============================================================
                # PARTE 3: Atualização com Modulação Geométrica
                # ============================================================
                
                # Passo do Adam: Δ_adam = m̂_t / √v̂_t
                step_adam = exp_avg_corrected / denom
                
                # Passo do AdamS: Δ_adamS = Δ_adam · σ
                step_adams = step_adam * sigma_final
                
                # Atualização: θ_{t+1} = θ_t - η·Δ_adamS
                p.data.add_(step_adams, alpha=-lr)
                
                # Salvar parâmetro anterior
                state['param_prev'] = theta_t.clone()
        
        # Atualizar loss global
        self.state['global_loss_prev'] = loss_t
        
        return loss


class AdamSW(AdamS):
    """
    AdamSW: AdamS com weight decay desacoplado (estilo AdamW).
    
    Aplica regularização L2 após a atualização do gradiente,
    independente da taxa de aprendizado.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 beta_s=0.9, alpha_min=0.5, alpha_max=1.5, 
                 warmup_steps=10, weight_decay=0.01):
        
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                        beta_s=beta_s, alpha_min=alpha_min, 
                        alpha_max=alpha_max, warmup_steps=warmup_steps)
        
        # Adicionar weight_decay aos defaults
        for group in self.param_groups:
            group['weight_decay'] = weight_decay
    
    @torch.no_grad()
    def step(self, closure=None, loss_item=None):
        """Passo com weight decay desacoplado."""
        
        # Executar passo normal do AdamS
        loss = super().step(closure=closure, loss_item=loss_item)
        
        # Aplicar weight decay APÓS atualização (desacoplado)
        for group in self.param_groups:
            if 'weight_decay' in group and group['weight_decay'] > 0:
                for p in group['params']:
                    if p.grad is not None:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])
        
        return loss