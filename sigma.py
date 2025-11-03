"""
SIGMA - Score-Informed Geometric Momentum Adaptation
Framework for stochastic optimization via a geometric convexity score in the loss function.

José Wilson C. Souza, 2025
"""

import torch
from torch.optim.optimizer import Optimizer

class SIGMA(Optimizer):
    """
    Implementa o otimizador SIGMA (Score-Informed Geometric Momentum Adaptation).
    
    Versão 1.3 - Com Momentum no Score
    
    Este otimizador utiliza interpolação geométrica entre iterações consecutivas
    para adaptar a taxa de aprendizado elemento-wise através de um "score" σ.
    
    Variantes:
    - SIGMA-T (Taylor): beta=None, usa score instantâneo
    - SIGMA-M (Momentum): beta ∈ [0,1), usa média móvel exponencial do score
    
    Args:
        params: Parâmetros iteráveis para otimizar
        lr (float): Taxa de aprendizado base η (default: 1e-2)
        beta (float, optional): Coeficiente de momentum para o score.
            - None: SIGMA-T (score instantâneo)
            - 0.0 ≤ beta < 1.0: SIGMA-M (score suavizado)
        alpha_min (float): Limite inferior para clipping do score σ (default: 0.1)
        alpha_max (float): Limite superior para clipping do score σ (default: 2.0)
        eps (float): Termo de estabilidade numérica (default: 1e-8)
    
    Referências:
        Baseado na análise de pontos de interseção C e D entre retas tangentes
        e suas propriedades de convexidade.
    """

    def __init__(self, params, lr=1e-2, beta=None, alpha_min=0.1, alpha_max=2.0, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")
            
        defaults = dict(
            lr=lr,
            beta=beta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps
        )
        super(SIGMA, self).__init__(params, defaults)
        
        # Estado global do otimizador
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, loss_item=None):
        """
        Executa um passo de otimização.
        
        IMPORTANTE: closure() (calculando loss.backward()) DEVE ser chamado ANTES.
        
        Args:
            loss_item (float): Valor escalar da loss na iteração atual.
                              OBRIGATÓRIO para o funcionamento do SIGMA.
        
        Returns:
            float: Valor da loss fornecida
            
        Raises:
            ValueError: Se loss_item não for fornecido
        """
        
        if loss_item is None:
            raise ValueError(
                "SIGMA.step() requer o argumento 'loss_item'. "
                "Exemplo: optimizer.step(loss_item=loss.item())"
            )
            
        loss_t = loss_item
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                g_t = p.grad.data
                theta_t = p.data
                state = self.state[p]

                # --- Inicialização do Estado ---
                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        # Inicializa momentum do score em 1.0 (neutro)
                        state['score_momentum'] = torch.ones_like(p.data)
                
                theta_prev = state['param_prev']
                
                # --- Cálculo do Score σ (baseado em interpolação geométrica) ---
                if self.state['global_step'] > 1:
                    # Pontos D (média harmônica) e C (média ponderada por f)
                    # D₁ = (θₜθₜ₋₁)/(θₜ + θₜ₋₁)
                    # D₂ = (θₜLₜ₋₁ + θₜ₋₁Lₜ)/(θₜ + θₜ₋₁)
                    
                    denom_xy = theta_prev + theta_t + eps
                    D1 = (theta_prev * theta_t) / denom_xy
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_xy
                    
                    # Aproximação de Taylor para f(2D₁)
                    f_proxy = loss_t + g_t * (2 * D1 - theta_t)
                    
                    # Score σ = D₂ / f(2D₁)
                    sigma_raw = D2 / (f_proxy + eps)
                    
                    # Tratamento de valores inválidos
                    sigma_raw[torch.isnan(sigma_raw)] = 1.0
                    sigma_raw[torch.isinf(sigma_raw)] = 1.0
                else:
                    # Primeira iteração: score neutro
                    sigma_raw = torch.ones_like(p.data)
                
                # --- Aplicação de Momentum (SIGMA-M) ---
                if beta is not None:
                    # Média móvel exponencial: S_t = β·S_{t-1} + (1-β)·σ_raw
                    score_mom = state['score_momentum']
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    # SIGMA-T: usa score instantâneo
                    sigma_to_use = sigma_raw
                
                # Clipping do score final
                sigma_to_use.clamp_(alpha_min, alpha_max)

                # --- Atualização dos Parâmetros ---
                # θₜ₊₁ = θₜ - η · gₜ ⊙ σₜ
                p.data.addcmul_(g_t, sigma_to_use, value=-lr)
                
                # Salvar parâmetros para próxima iteração
                state['param_prev'] = theta_t.clone()
        
        # Atualizar loss global
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        """Representação string do otimizador."""
        variant = "SIGMA-M" if self.defaults['beta'] is not None else "SIGMA-T"
        return f"{variant}(lr={self.defaults['lr']}, beta={self.defaults['beta']})"