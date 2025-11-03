"""
AdaSIGMA: Adaptive Moment Estimation with Symmetric Interpolation Geometry
===========================================================================

Fusão matemática do Adam com SIGMA, integrando o score geométrico σ 
na regra de atualização adaptativa de momentos.

Fundamentação Matemática:
-------------------------
Adam usa:
    m_t = β₁·m_{t-1} + (1-β₁)·g_t          (momento de 1ª ordem)
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²         (momento de 2ª ordem)
    θ_{t+1} = θ_t - α·m̂_t/√(v̂_t + ε)

SIGMA usa:
    σ_t = D₂/(L_t + g_t·(2D₁ - θ_t) + ε)  (score geométrico)
    θ_{t+1} = θ_t - η·g_t⊙σ_t

AdaSIGMA propõe:
    θ_{t+1} = θ_t - α·(m̂_t⊙σ_t)/√(v̂_t + ε)

Onde σ_t modula o momento do gradiente, criando uma adaptação geométrica
adicional sobre a taxa de aprendizado adaptativa do Adam.

Referências:
    - Kingma & Ba (2015): Adam optimizer
    - Souza (2025): SIGMA geometric interpolation
"""

import torch
from torch.optim.optimizer import Optimizer


class AdaSIGMA(Optimizer):
    """
    AdaSIGMA: Fusão do Adam com SIGMA via modulação geométrica do momento.
    
    A regra de atualização integra:
    1. Momentos adaptativos do Adam (m_t, v_t)
    2. Score geométrico σ_t do SIGMA
    3. Bias correction para ambos
    
    Args:
        params: Parâmetros iteráveis para otimizar
        lr (float): Taxa de aprendizado (default: 1e-3)
        betas (tuple): Coeficientes para médias móveis de gradiente e seu quadrado (default: (0.9, 0.999))
        beta_sigma (float): Coeficiente de momentum para o score σ (default: 0.9)
        alpha_min (float): Limite inferior para σ (default: 0.1)
        alpha_max (float): Limite superior para σ (default: 2.0)
        eps (float): Termo de estabilidade numérica (default: 1e-8)
        weight_decay (float): Regularização L2 (default: 0)
        amsgrad (bool): Usar variante AMSGrad (default: False)
    
    Exemplo:
        >>> optimizer = AdaSIGMA(model.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step(loss_item=loss.item())
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_sigma=0.9,
                 alpha_min=0.1, alpha_max=2.0, eps=1e-8, weight_decay=0, amsgrad=False,
                 sparse_ratio=1.0):
        """
        Args:
            sparse_ratio: Fração de parâmetros para calcular σ (default: 1.0 = 100%)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= beta_sigma < 1.0:
            raise ValueError(f"Invalid beta_sigma: {beta_sigma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Invalid alpha bounds: [{alpha_min}, {alpha_max}]")
        if not 0.0 < sparse_ratio <= 1.0:
            raise ValueError(f"Invalid sparse_ratio: {sparse_ratio}")

        defaults = dict(lr=lr, betas=betas, beta_sigma=beta_sigma,
                        alpha_min=alpha_min, alpha_max=alpha_max,
                        eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        sparse_ratio=sparse_ratio)
        super(AdaSIGMA, self).__init__(params, defaults)
        
        # Estado global
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, closure=None, loss_item=None):
        """
        Executa um passo de otimização.
        
        Args:
            closure: Um closure que reavalia o modelo e retorna a loss (opcional)
            loss_item (float): Valor escalar da loss (obrigatório para AdaSIGMA)
        
        Returns:
            loss: Loss se closure foi fornecido, caso contrário None
        """
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if loss_item is None:
            raise ValueError(
                "AdaSIGMA.step() requer 'loss_item'. "
                "Use: optimizer.step(loss_item=loss.item())"
            )
        
        loss_t = loss_item
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta_sigma = group['beta_sigma']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            sparse_ratio = group['sparse_ratio']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaSIGMA does not support sparse gradients')
                
                amsgrad = group['amsgrad']
                
                state = self.state[p]
                
                # Inicialização do estado
                if len(state) == 0:
                    state['step'] = 0
                    # Momentos do Adam
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Para SIGMA
                    state['param_prev'] = p.data.clone()
                    state['score_momentum'] = torch.ones_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                state['step'] += 1
                
                # Weight decay (regularização L2)
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # ============================================================
                # PARTE 1: Atualização dos Momentos (Adam)
                # ============================================================
                
                # m_t = β₁·m_{t-1} + (1-β₁)·g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # v_t = β₂·v_{t-1} + (1-β₂)·g_t²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    # v̂_t = max(v̂_{t-1}, v_t)
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                # ============================================================
                # PARTE 2: Cálculo do Score Geométrico σ (SIGMA)
                # ============================================================
                
                theta_prev = state['param_prev']
                theta_t = p.data
                
                # Decisão de amostragem esparsa
                if sparse_ratio < 1.0:
                    sample_mask = torch.rand_like(theta_t) < sparse_ratio
                    compute_sigma = sample_mask.any().item()
                else:
                    compute_sigma = True
                    sample_mask = None
                
                if self.state['global_step'] > 1 and compute_sigma:
                    # Pontos D (média harmônica e ponderada)
                    denom_geom = theta_prev + theta_t + group['eps']
                    D1 = (theta_prev * theta_t) / denom_geom
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_geom
                    
                    # Aproximação de Taylor para f(2D₁)
                    f_proxy = loss_t + grad * (2 * D1 - theta_t)
                    
                    # Score σ_raw = D₂ / f_proxy
                    sigma_raw = D2 / (f_proxy + group['eps'])
                    
                    # Tratamento de valores inválidos
                    sigma_raw = torch.where(
                        torch.isfinite(sigma_raw),
                        sigma_raw,
                        torch.ones_like(sigma_raw)
                    )
                    
                    # Se esparso, aplicar σ apenas nos elementos amostrados
                    if sample_mask is not None:
                        sigma_raw = torch.where(sample_mask, sigma_raw, torch.ones_like(sigma_raw))
                else:
                    sigma_raw = torch.ones_like(p.data)
                
                # Aplicar momentum no score
                if beta_sigma is not None and beta_sigma > 0:
                    score_mom = state['score_momentum']
                    score_mom.mul_(beta_sigma).add_(sigma_raw, alpha=1 - beta_sigma)
                    sigma = score_mom
                else:
                    sigma = sigma_raw
                
                # Clipping do score
                sigma = torch.clamp(sigma, alpha_min, alpha_max)
                
                # ============================================================
                # PARTE 3: Atualização Híbrida (Fusão)
                # ============================================================
                
                # AdaSIGMA: θ_{t+1} = θ_t - α·(m̂_t⊙σ_t)/√(v̂_t + ε)
                # 
                # Interpretação:
                # - m̂_t/√(v̂_t): direção adaptativa do Adam
                # - σ_t: modulação geométrica do SIGMA
                # - Produto elemento-wise combina ambas as adaptações
                
                p.data.addcdiv_(exp_avg * sigma, denom, value=-step_size)
                
                # Salvar estado para próxima iteração
                state['param_prev'] = theta_t.clone()
        
        # Atualizar loss global
        self.state['global_loss_prev'] = loss_t
        
        return loss
    
    def __repr__(self):
        """Representação string do otimizador."""
        return (f"AdaSIGMA(lr={self.defaults['lr']}, "
                f"betas={self.defaults['betas']}, "
                f"beta_sigma={self.defaults['beta_sigma']})")


class AdaSIGMAW(AdaSIGMA):
    """
    AdaSIGMAW: AdaSIGMA com weight decay decoupled (estilo AdamW).
    
    Implementa weight decay desacoplado conforme Loshchilov & Hutter (2019),
    que aplica a regularização diretamente aos parâmetros ao invés de 
    adicioná-la ao gradiente.
    
    Args:
        Mesmos argumentos do AdaSIGMA
    
    Referência:
        Loshchilov & Hutter (2019): Decoupled Weight Decay Regularization
    """
    
    @torch.no_grad()
    def step(self, closure=None, loss_item=None):
        """Passo de otimização com weight decay desacoplado."""
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if loss_item is None:
            raise ValueError("AdaSIGMAW.step() requer 'loss_item'")
        
        loss_t = loss_item
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta_sigma = group['beta_sigma']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                
                # Inicialização
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['param_prev'] = p.data.clone()
                    state['score_momentum'] = torch.ones_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Momentos do Adam (SEM adicionar weight decay ao gradiente)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                # Cálculo do score σ (idêntico ao AdaSIGMA)
                theta_prev = state['param_prev']
                theta_t = p.data
                
                # Decisão de amostragem esparsa
                if sparse_ratio < 1.0:
                    sample_mask = torch.rand_like(theta_t) < sparse_ratio
                    compute_sigma = sample_mask.any().item()
                else:
                    compute_sigma = True
                    sample_mask = None
                
                if self.state['global_step'] > 1 and compute_sigma:
                    denom_geom = theta_prev + theta_t + group['eps']
                    D1 = (theta_prev * theta_t) / denom_geom
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_geom
                    f_proxy = loss_t + grad * (2 * D1 - theta_t)
                    sigma_raw = D2 / (f_proxy + group['eps'])
                    sigma_raw = torch.where(
                        torch.isfinite(sigma_raw),
                        sigma_raw,
                        torch.ones_like(sigma_raw)
                    )
                else:
                    sigma_raw = torch.ones_like(p.data)
                
                if beta_sigma is not None and beta_sigma > 0:
                    score_mom = state['score_momentum']
                    score_mom.mul_(beta_sigma).add_(sigma_raw, alpha=1 - beta_sigma)
                    sigma = score_mom
                else:
                    sigma = sigma_raw
                
                sigma = torch.clamp(sigma, alpha_min, alpha_max)
                
                # Atualização com weight decay DESACOPLADO
                # θ_{t+1} = θ_t - α·(m̂_t⊙σ_t)/√(v̂_t) - λ·α·θ_t
                p.data.addcdiv_(exp_avg * sigma, denom, value=-step_size)
                
                # Weight decay desacoplado (aplicado após a atualização)
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                state['param_prev'] = theta_t.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss