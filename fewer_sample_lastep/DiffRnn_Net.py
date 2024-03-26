import torch
import torch.nn as nn
import math

# Diffusion rnn network
class DRNet(nn.Module):
    def __init__(self):
        super(DRNet, self).__init__()
        
        # hidden layer to previous timestep
        self.h2h = nn.Sequential(
            nn.Linear(256, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 256),
        )

    def forward(self, T, g_cumsum, g_pixel_tensor, loss_fn, L):

        # t ~ Uniform ({1, ...T})
        timestep = torch.arange(1, T+1, 1)
        # Initialize y_0 
        latent_y_0 = g_cumsum[:, :, T-1]

        for t in timestep:
            # set eq
            a_t = (t+1)/(T+1)
            a_t_minus = t/(T+1)
            a_T = (T+1)/(T+1)
            t_prime = T-t 
            sigma_t = math.sqrt(a_t*(1-a_t))

            # epsilon ~ N(0, I) * 1e-2
            # epsilon = 1e-2 * torch.randn_like(latent_y_0)
            epsilon = 1e-1 * torch.randn_like(latent_y_0)
            
            # locate the current y_t by cumulated sum of g_net output with added noise
            y_t = latent_y_0 - a_t/a_T * g_cumsum[:, :, t_prime-1] + sigma_t*epsilon
            # print(y_t)
            # calculate expected y_t
            expected_y_t_minus = latent_y_0 - a_t_minus/a_T * g_cumsum[:, :, t_prime]

            # locate the g(x) at current timepoint
            x_point = g_pixel_tensor[:, :, t_prime-1].view(g_pixel_tensor.size(0), -1)
            # learn link from sample in the current timepoint sphere to the previous point
            input = y_t + x_point
            out = self.h2h(input)

            # training loss
            # lambda*||f, expected_y_(t-1) - (y_t + g(x))||
            loss = loss_fn(out, expected_y_t_minus - (y_t + x_point))
            lambda_t = 1/a_t_minus - 1/a_t
            L += lambda_t * loss
        
        return out, L
    
class DRNetTest(nn.Module):
    def __init__(self):
        super(DRNetTest, self).__init__()

        self.h2h = nn.Sequential(
            nn.Linear(256, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 256),
        )

    def forward(self, T, g_pixel_tensor):
        latent_y_T = torch.zeros(g_pixel_tensor.size(0), 256).to(device)

        # t ~ Uniform ({1, ...T})
        timestep = torch.arange(T, 0, -1)
        # timestep = torch.arange(1, T+1, 1)

        for t in timestep:
            # set eq
            a_t = (t+1)/(T+1)
            t_prime = T-t 
            sigma_t = math.sqrt(a_t*(1-a_t))

            # epsilon ~ N(0, I) * 1e-2
            epsilon = 1e-1 * torch.randn_like(latent_y_T)
            
            # locate the g(x) at current timepoint
            x_point = g_pixel_tensor[:, :, t_prime].view(g_pixel_tensor.size(0), -1)
            # predict the link to embedding of next timestep
            input = latent_y_T + x_point
            out = self.h2h(input)
            # calculate the location of embedding of next timestep
            latent_y_T = latent_y_T + x_point + out + sigma_t*epsilon

        return latent_y_T