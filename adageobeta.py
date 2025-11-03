import torch
from torch.optim.optimizer import Optimizer
import math

class AdamGeoBeta(Optimizer):
    """
    Implementa o Otimizador Adam-Geo-Beta.
    
    Proposta 3 (a mais avançada): O Adam completo, mas o score 
    geométrico (Log-U)
    é usado para modular dinamicamente o hiperparâmetro beta_2.
    
    beta_2_dinamico = clamp(beta_2_base * sigma_final, 0.0, 0.9999)
    v_t = beta_2_dinamico * v_{t-1} + (1 - beta_2_dinamico) * g_t^2
    """

    def __init__(self, params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 beta_s=0.9, # Usando o 'Fast Brake' como padrão
                 alpha_min=0.1, 
                 alpha_max=2.0,
                 warmup_steps=0):
        
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Beta 1 (grad) inválido: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Beta 2 (var base) inválido: {betas[1]}")
        if not 0.0 <= beta_s < 1.0:
            raise ValueError(f"Beta_s (score) inválido: {beta_s}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        beta_s=beta_s, alpha_min=alpha_min, 
                        alpha_max=alpha_max, warmup_steps=warmup_steps)
        
        super(AdamGeoBeta, self).__init__(params, defaults)
        
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, loss_item=None):
        if loss_item is None:
            raise ValueError("AdamGeoBeta.step() requer um argumento 'loss_item'.")
            
        loss_t = loss_item
        self.state['global_step'] += 1
        step = self.state['global_step']
        loss_prev = self.state['global_loss_prev']
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2_base = group['betas'] # beta2 é o "base"
            eps = group['eps']
            beta_s = group['beta_s']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            warmup_steps = group['warmup_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                g_t = p.grad.data
                theta_t = p.data
                state = self.state[p]

                # --- Inicialização do Estado ---
                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    state['exp_avg'] = torch.zeros_like(p.data) # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p.data) # v_t
                    state['log_score_momentum'] = torch.zeros_like(p.data) # m_L_U
                
                theta_prev = state['param_prev']
                m_t = state['exp_avg']
                v_t = state['exp_avg_sq']
                m_L_U = state['log_score_momentum']

                # --- 1. Cálculo do Score SIGMA (Log-U) ---
                if step <= warmup_steps or step <= 1:
                    log_score_raw = torch.zeros_like(p.data) # log(1.0) = 0.0
                else:
                    # Score D (Teorema 1)
                    denom_xy = theta_prev + theta_t + eps
                    D1 = (theta_prev * theta_t) / denom_xy
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_xy
                    f_proxy_D = loss_t + g_t * (2 * D1 - theta_t)
                    sigma_D_raw = (D2 / (f_proxy_D + eps)).clamp_(alpha_min, alpha_max)
                    
                    # Score C (Teorema 2)
                    denom_fxfy = loss_prev + loss_t + eps
                    C1 = (loss_prev * theta_t + loss_t * theta_prev) / denom_fxfy
                    C2 = (loss_prev * loss_t) / denom_fxfy
                    f_proxy_C = loss_t + g_t * (C1 - theta_t)
                    sigma_C_raw = ((2 * C2) / (f_proxy_C + eps)).clamp_(alpha_min, alpha_max)

                    # Score U (Log-Space)
                    log_score_D = torch.log(sigma_D_raw + eps)
                    log_score_C = torch.log(sigma_C_raw + eps)
                    log_score_raw = log_score_C + log_score_D
                    log_score_raw[torch.isnan(log_score_raw)] = 0.0
                    log_score_raw[torch.isinf(log_score_raw)] = 0.0
                
                # Momentum em Log-Space
                m_L_U.mul_(beta_s).add_(log_score_raw, alpha=1 - beta_s)
                
                # Converter de volta
                sigma_final = torch.exp(m_L_U).clamp_(alpha_min, alpha_max)

                # --- 2. Cálculo dos Momentos Adam ---
                
                # Cálculo do m_t (padrão)
                m_t.mul_(beta1).add_(g_t, alpha=1 - beta1)
                
                # --- A INOVAÇÃO (Adam-Geo-Beta) ---
                beta2_dynamic = (beta2_base * sigma_final).clamp_(0.0, 0.9999)
                
                # --- CORREÇÃO (v1.1) ---
                # A linha original `v_t.mul_(...).addcmul_(...)` falha
                # porque 'value' não pode ser um Tensor.
                
                # 1. v_t = beta_2_dinamico * v_{t-1}
                v_t.mul_(beta2_dynamic)
                
                # 2. v_t += (1 - beta_2_dinamico) * g_t^2
                g_t_sq = g_t.pow(2)
                one_minus_beta2_dyn = 1.0 - beta2_dynamic
                
                # Usar addcmul_ com tensores e value=1.0 (escalar)
                # v_t = v_t + 1.0 * (g_t_sq * one_minus_beta2_dyn)
                v_t.addcmul_(g_t_sq, one_minus_beta2_dyn, value=1.0)
                # --- FIM DA CORREÇÃO ---

                bias_correction1 = 1 - beta1 ** step
                bias_correction2_dynamic = 1 - beta2_dynamic ** step
                
                m_t_hat = m_t / bias_correction1
                # Evitar divisão por zero se bias_correction for 0
                v_t_hat = v_t / (bias_correction2_dynamic + eps) 
                
                denom = v_t_hat.sqrt().add_(eps)

                # --- 3. Regra de Atualização (Adam Padrão) ---
                p.data.addcdiv_(m_t_hat, denom, value=-lr)
                
                # --- 4. Salvar estado ---
                state['param_prev'] = theta_t.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t