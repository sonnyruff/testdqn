import torch
from torch import nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """
    def __init__(self, in_features: int, out_features: int, distr_type: str = 'normal', std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.distr_type = distr_type
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.resample_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def resample_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features, self.distr_type)
        epsilon_out = self.scale_noise(self.out_features, self.distr_type)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu,
            self.bias_mu,
        )
    
    def get_noise(self):
        return {
            "weight_epsilon": self.weight_epsilon.clone().detach().cpu().numpy(),
            "bias_epsilon": self.bias_epsilon.clone().detach().cpu().numpy(),
        }
    
    @staticmethod
    def scale_noise(size: int, distr_type: str) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        if distr_type == 'normal':
            x = torch.randn(size)
        elif distr_type == 'uniform':
            x = torch.rand(size)

        return x.sign().mul(x.abs().sqrt())


    

if __name__ == '__main__':
    noisy = NoisyLinear(2, 3)
    print(noisy(torch.rand(1, 2)))