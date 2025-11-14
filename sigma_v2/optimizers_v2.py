"""
Copyright 2025 José Wilson C. Souza

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUTHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Otimizadores SIGMA v2.0 - Implementações Avançadas

Contém:
- SIGMA_D_v2: Otimizador com score do Teorema 1 (Ponto D),
              baseado em sigma_v2.py.
- SIGMA_C_v2: Otimizador com score do Teorema 2 (Ponto C),
              adaptado com os mesmos recursos avançados.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable
import math 

# ============================================================================
# OTIMIZADOR 1: SIGMA_D_v2 (Score Teorema 1 - Ponto D)
# ============================================================================

class SIGMA_D_v2(Optimizer):
    """
    Implementa o otimizador SIGMA (Score-Informed Geometric Momentum Adaptation).
    Versão 2.0 - Baseada no Teorema 1 (Ponto D)
    
    Possui recursos modernos: warmup, weight decay desacoplado, grad_clip,
    amsgrad, e aproximação de 2ª ordem.
    """

    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        beta: Optional[float] = None,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
        warmup_steps: int = 0,
        second_order: bool = False,
        amsgrad: bool = False
    ):
        # Validação de parâmetros
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Weight decay inválido: {weight_decay}")
        if grad_clip is not None and not grad_clip > 0:
            raise ValueError(f"Gradient clipping inválido: {grad_clip}")
        if not warmup_steps >= 0:
            raise ValueError(f"Warmup steps inválido: {warmup_steps}")
            
        defaults = dict(
            lr=lr,
            beta=beta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            warmup_steps=warmup_steps,
            second_order=second_order,
            amsgrad=amsgrad
        )
        super(SIGMA_D_v2, self).__init__(params, defaults)
        
        # Estado global do otimizador
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    def _get_lr_with_warmup(self, base_lr: float) -> float:
        """Aplica warmup linear na taxa de aprendizado."""
        step = self.state['global_step']
        warmup = self.defaults['warmup_steps']
        
        if warmup == 0 or step >= warmup:
            return base_lr
        else:
            return base_lr * (step + 1) / warmup

    @torch.no_grad()
    def step(self, loss_item: Optional[float] = None, closure: Optional[Callable] = None):
        if loss_item is None:
            raise ValueError(
                "SIGMA.step() requer o argumento 'loss_item'. "
                "Exemplo: optimizer.step(loss_item=loss.item())"
            )
        
        # Atualizar estado global
        loss_t = float(loss_item)
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        step = self.state['global_step']
        
        for group in self.param_groups:
            base_lr = group['lr']
            lr = self._get_lr_with_warmup(base_lr)
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']
            weight_decay = group['weight_decay']
            grad_clip = group['grad_clip']
            second_order = group['second_order']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if grad_clip is not None:
                    grad.clamp_(-grad_clip, grad_clip)
                
                theta_t = p.data
                state = self.state[p]

                # --- Inicialização do Estado ---
                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        state['score_momentum'] = torch.ones_like(p.data)
                    if second_order:
                        state['grad_prev'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_score'] = torch.ones_like(p.data)
                
                theta_prev = state['param_prev']
                
                # --- Cálculo do Score σ (Teorema 1 - Ponto D) ---
                if step > 1:
                    denom_xy = torch.abs(theta_prev) + torch.abs(theta_t) + eps
                    D1 = (theta_prev * theta_t) / denom_xy
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_xy
                    
                    if second_order and 'grad_prev' in state:
                        grad_prev = state['grad_prev']
                        delta_theta = theta_t - theta_prev
                        delta_grad = grad - grad_prev
                        hess_approx = delta_grad / (delta_theta + eps)
                        diff = 2 * D1 - theta_t
                        f_proxy = loss_t + (grad * diff).sum().item()
                        f_proxy += 0.5 * (diff * hess_approx * diff).sum().item()
                        f_proxy = max(f_proxy, eps)
                    else:
                        f_proxy = loss_t + grad * (2 * D1 - theta_t)
                        f_proxy = torch.clamp(f_proxy, min=eps)
                    
                    sigma_raw = D2 / (f_proxy + eps)
                    
                    sigma_raw = torch.where(
                        torch.isnan(sigma_raw) | torch.isinf(sigma_raw),
                        torch.ones_like(sigma_raw),
                        sigma_raw
                    )
                else:
                    sigma_raw = torch.ones_like(p.data)
                
                # --- Aplicação de Momentum (SIGMA-M) ---
                if beta is not None:
                    score_mom = state['score_momentum']
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw
                
                # --- AMSGrad Variant (opcional) ---
                if amsgrad:
                    max_score = state['max_score']
                    torch.max(max_score, torch.abs(sigma_to_use), out=max_score)
                    sigma_to_use = max_score
                
                sigma_to_use = torch.clamp(sigma_to_use, alpha_min, alpha_max)

                # --- Atualização dos Parâmetros ---
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                p.data.addcmul_(grad, sigma_to_use, value=-lr)
                
                state['param_prev'] = theta_t.clone()
                if second_order:
                    state['grad_prev'] = grad.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        return f"SIGMA-D_v2(lr={self.defaults['lr']}, beta={self.defaults['beta']})"

# ============================================================================
# OTIMIZADOR 2: SIGMA_C_v2 (Score Teorema 2 - Ponto C)
# ============================================================================

class SIGMA_C_v2(Optimizer):
    """
    Implementa o otimizador SIGMA (Score-Informed Geometric Momentum Adaptation).
    Versão 2.0 - Baseada no Teorema 2 (Ponto C)
    
    Possui os mesmos recursos avançados do SIGMA_D_v2 (warmup, weight decay,
    grad_clip, amsgrad, 2ª ordem), mas usa a formulação de 
    score do Teorema 2.
    """

    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        beta: Optional[float] = None,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
        warmup_steps: int = 0,
        second_order: bool = False,
        amsgrad: bool = False
    ):
        # Validação de parâmetros (idêntica ao v2)
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Weight decay inválido: {weight_decay}")
        if grad_clip is not None and not grad_clip > 0:
            raise ValueError(f"Gradient clipping inválido: {grad_clip}")
        if not warmup_steps >= 0:
            raise ValueError(f"Warmup steps inválido: {warmup_steps}")
            
        defaults = dict(
            lr=lr,
            beta=beta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            warmup_steps=warmup_steps,
            second_order=second_order,
            amsgrad=amsgrad
        )
        super(SIGMA_C_v2, self).__init__(params, defaults)
        
        # Estado global do otimizador
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    def _get_lr_with_warmup(self, base_lr: float) -> float:
        """Aplica warmup linear na taxa de aprendizado."""
        step = self.state['global_step']
        warmup = self.defaults['warmup_steps']
        
        if warmup == 0 or step >= warmup:
            return base_lr
        else:
            return base_lr * (step + 1) / warmup

    @torch.no_grad()
    def step(self, loss_item: Optional[float] = None, closure: Optional[Callable] = None):
        if loss_item is None:
            raise ValueError(
                "SIGMA.step() requer o argumento 'loss_item'. "
                "Exemplo: optimizer.step(loss_item=loss.item())"
            )
        
        # Atualizar estado global
        loss_t = float(loss_item)
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        step = self.state['global_step']
        
        for group in self.param_groups:
            base_lr = group['lr']
            lr = self._get_lr_with_warmup(base_lr)
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']
            weight_decay = group['weight_decay']
            grad_clip = group['grad_clip']
            second_order = group['second_order']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if grad_clip is not None:
                    grad.clamp_(-grad_clip, grad_clip)
                
                theta_t = p.data
                state = self.state[p]

                # --- Inicialização do Estado ---
                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        state['score_momentum'] = torch.ones_like(p.data)
                    if second_order:
                        state['grad_prev'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_score'] = torch.ones_like(p.data)
                
                theta_prev = state['param_prev']
                
                # --- Cálculo do Score σ (Teorema 2 - Ponto C) ---
                if step > 1:
                    
                    # --- [MODIFICAÇÃO 2] CÁLCULO DO C1_INTEGRAL (E_GLOBAL) ---
                    # Substitui o cálculo original de C1 pela fórmula 
                    # analítica da E_global para a geometria L2 (f(x)=x^2).
                    
                    a = theta_prev
                    b = theta_t

                    # 1. Lidar com o caso degenerado (a ≈ b)
                    # Se a e b são muito próximos, a média é apenas a si mesma
                    diff_sq = (b - a).pow(2)
                    is_degenerate = diff_sq < (eps**2) # Usar eps^2
                    
                    # 2. Calcular a fórmula analítica (caso não-degenerado)
                    # Criar cópias seguras para evitar 0 em log/divisão
                    a_safe = a.clone()
                    b_safe = b.clone()
                    # Garante que valores muito pequenos não sejam 0
                    a_safe[a_safe.abs() < eps] = eps * torch.sign(a_safe[a_safe.abs() < eps]) + eps
                    b_safe[b_safe.abs() < eps] = eps * torch.sign(b_safe[b_safe.abs() < eps]) + eps

                    a2_plus_b2_safe = a.pow(2) + b.pow(2) + eps
                    a3 = a.pow(3)
                    b3 = b.pow(3)
                    
                    # Termo 1: T1 = 2(a+b)/3
                    Term1_Eglobal = (2.0 * (a + b) / 3.0)
                    
                    # Denominador da fração: 3(b-a)^2 + eps
                    denom_frac = 3.0 * diff_sq + eps
                    
                    # Numerador da fração (T2 + T3 + T4)
                    
                    # T2 = (a^3+b^3) * (log(2 / (a^2+b^2+eps)) - pi/2)
                    T2_num = (a3 + b3) * (torch.log(2.0 / a2_plus_b2_safe) - (math.pi / 2.0))
                    
                    # T3 = 2 * (a^3 * log(|a|) + b^3 * log(|b|))
                    T3_num = 2.0 * (a3 * torch.log(torch.abs(a_safe)) + b3 * torch.log(torch.abs(b_safe)))
                    
                    # T4 = 2 * (a^3 * atan(b/a) + b^3 * atan(a/b))
                    T4_num = 2.0 * (a3 * torch.atan(b_safe / a_safe) + b3 * torch.atan(a_safe / b_safe))
                    
                    numerator_frac = T2_num + T3_num + T4_num
                    
                    # E_global (caso não degenerado)
                    E_global_formula = Term1_Eglobal + (numerator_frac / denom_frac)
                    
                    # 3. Combinar os casos
                    C1 = torch.where(
                        is_degenerate,
                        (a + b) / 2.0, # Se a ≈ b, a média é a média aritmética
                        E_global_formula
                    )
                    
                    # 4. Limpeza final de NaN/Inf (fallback seguro)
                    C1 = torch.where(
                        torch.isnan(C1) | torch.isinf(C1),
                        (a + b) / 2.0,
                        C1
                    )
                    # --- FIM DA MODIFICAÇÃO ---

                    # Cálculo do two_C2 (permanece o mesmo)
                    denom_L = abs(loss_prev) + abs(loss_t) + eps
                    two_C2 = (2 * loss_t * loss_prev) / denom_L
                    
                    if second_order and 'grad_prev' in state:
                        grad_prev = state['grad_prev']
                        delta_theta = theta_t - theta_prev
                        delta_grad = grad - grad_prev
                        hess_approx = delta_grad / (delta_theta + eps)
                        
                        diff = C1 - theta_t
                        f_proxy = loss_t + (grad * diff).sum().item()
                        f_proxy += 0.5 * (diff * hess_approx * diff).sum().item()
                        f_proxy = max(f_proxy, eps)
                    else:
                        f_proxy = loss_t + grad * (C1 - theta_t)
                        f_proxy = torch.clamp(f_proxy, min=eps)
                    
                    sigma_raw = two_C2 / (f_proxy + eps)
                    
                    sigma_raw = torch.where(
                        torch.isnan(sigma_raw) | torch.isinf(sigma_raw),
                        torch.ones_like(sigma_raw),
                        sigma_raw
                    )
                else:
                    sigma_raw = torch.ones_like(p.data)
                
                # --- Aplicação de Momentum (SIGMA-M) ---
                if beta is not None:
                    score_mom = state['score_momentum']
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw
                
                # --- AMSGrad Variant (opcional) ---
                if amsgrad:
                    max_score = state['max_score']
                    torch.max(max_score, torch.abs(sigma_to_use), out=max_score)
                    sigma_to_use = max_score
                
                sigma_to_use = torch.clamp(sigma_to_use, alpha_min, alpha_max)

                # --- Atualização dos Parâmetros ---
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                p.data.addcmul_(grad, sigma_to_use, value=-lr)
                
                state['param_prev'] = theta_t.clone()
                if second_order:
                    state['grad_prev'] = grad.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        return f"SIGMA-C_v2(lr={self.defaults['lr']}, beta={self.defaults['beta']})"