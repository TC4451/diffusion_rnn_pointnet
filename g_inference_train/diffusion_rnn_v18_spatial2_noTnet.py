import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import math
import matplotlib as plt
import logging
import numpy as np

# load MNIST dataset, convert to binary pixel values
mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: torch.where(x > 0,1,0))
                             ]))
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=False)
mnist_test = datasets.MNIST(root='./data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: torch.where(x > 0,1,0))
                             ]))
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
# set device to run
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # encoder model  
# class TransformationNet(nn.Module):

#     def __init__(self, input_dim, output_dim):
#         super(TransformationNet, self).__init__()
#         self.output_dim = output_dim

#         self.conv_1 = nn.Conv1d(input_dim, 64, 1)
#         self.conv_2 = nn.Conv1d(64, 128, 1)
#         self.conv_3 = nn.Conv1d(128, 256, 1)

#         self.bn_1 = nn.BatchNorm1d(64)
#         self.bn_2 = nn.BatchNorm1d(128)
#         self.bn_3 = nn.BatchNorm1d(256)
#         self.bn_4 = nn.BatchNorm1d(256)
#         self.bn_5 = nn.BatchNorm1d(128)

#         self.fc_1 = nn.Linear(256, 256)
#         self.fc_2 = nn.Linear(256, 128)
#         self.fc_3 = nn.Linear(128, self.output_dim*self.output_dim)

#     def forward(self, x):
#         num_points = x.shape[1]
#         x = x.transpose(2, 1)
#         x = F.relu(self.bn_1(self.conv_1(x)))
#         x = F.relu(self.bn_2(self.conv_2(x)))
#         x = F.relu(self.bn_3(self.conv_3(x)))

#         x = nn.MaxPool1d(num_points)(x)
#         x = x.view(-1, 256)

#         x = F.relu(self.bn_4(self.fc_1(x)))
#         x = F.relu(self.bn_5(self.fc_2(x)))
#         x = self.fc_3(x)

#         identity_matrix = torch.eye(self.output_dim)
#         if torch.cuda.is_available():
#             identity_matrix = identity_matrix.cuda()
#         x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
#         return x


class BasePointNet(nn.Module):

    def __init__(self, point_dimension):
        super(BasePointNet, self).__init__()
        # print(f"____point dim:{point_dimension}")
        # self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        # self.feature_transform = TransformationNet(input_dim=64, output_dim=64)
        
        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(256)
        

    def forward(self, x, plot=False):
        num_points = x.shape[1]
        
        # input_transform = self.input_transform(x) # T-Net tensor [batch, 3, 3]
        # x = torch.bmm(x, input_transform) # Batch matrix-matrix product 
        x = x.transpose(2, 1) 
        # tnet_out=x.cpu().detach().numpy()
        
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        # x = x.transpose(2, 1)

        # feature_transform = self.feature_transform(x) # T-Net tensor [batch, 64, 64]
        # x = torch.bmm(x, feature_transform)
        # x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x))) 
        # x = torch.sum(x, dim=2)  # summation
        # x = x.view(-1, 256)  # global feature vector 

        return x
    
def img_block_pos_expand(img_block, base):
    dim = img_block.shape
    values = img_block.view(dim[0], -1, 1)
    spacial_encoding_mat = torch.from_numpy(get_pos_matrix(values.shape[1], base_num=base)).T
    spacial_encoding_mat = spacial_encoding_mat.expand(dim[0], spacial_encoding_mat.shape[0], spacial_encoding_mat.shape[1])
    pc = torch.cat((spacial_encoding_mat, values), dim=2)
    return pc

def get_pos_matrix(n, base_num):
    # get the number of binary digits to represent n
    dim = int(np.ceil(np.log(n) / np.log(base_num)))

    mat = np.zeros((dim, base_num**dim), dtype=int)
    # basic pattern
    base = np.array([range(base_num)])

    for ii in range(dim):
        # number of same numbers in a row (repeat along row)
        unit = np.repeat(base, base_num**(ii), axis=1)
        # number of repeats (repeat along column)
        full = np.repeat(unit, base_num**(dim-ii-1), axis=0)
        mat[ii] = full.flatten()
    norm_mat = mat / base_num  # will give error from 0/0, but not important
    return norm_mat[:, 1:n+1]

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

    def forward(self, T, g_cumsum, g_pixel_tensor, loss_fn, L, g_inf_tensor):

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
            epsilon = 1e-1 * torch.randn_like(latent_y_0)
            # epsilon = torch.randn_like(latent_y_0)
            
            # locate the current y_t by cumulated sum of g_net output with added noise
            y_t = latent_y_0 - a_t/a_T * g_cumsum[:, :, t_prime-1] + sigma_t*epsilon
            # calculate expected y_t
            expected_y_t_minus = latent_y_0 - a_t_minus/a_T * g_cumsum[:, :, t_prime]

            # locate the g(x) at current timepoint
            x_point = g_pixel_tensor[:, :, t_prime-1].view(g_pixel_tensor.size(0), -1)
            learned_x_point = g_inf_tensor[:, :, t_prime-1].view(g_inf_tensor.size(0), -1)
            # learn link from sample in the current timepoint sphere to the previous point
            input = y_t + learned_x_point
            out = self.h2h(input)

            # training loss
            # lambda*||f, expected_y_(t-1) - (y_t + g(x))||
            loss = loss_fn(out, expected_y_t_minus - (y_t + x_point))
            lambda_t = 1/a_t_minus - 1/a_t
            L += lambda_t * loss
        
        return out, L

# Train function for f
def train_f(point_net, params):
    num_classes = params['num_classes']
    lr = params['lr_f']
    num_epochs = params['num_epochs']
    T = params['T']
    # Initialize network
    f_net = DRNet().to(device)
    g_inf_net = BasePointNet(point_dimension=params['point_dimension']).to(device)
    optimizer = torch.optim.AdamW(f_net.parameters(), lr)
    loss_fn = torch.nn.MSELoss().to(device)
    for e in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            # initialize y_0
            y_0 = torch.tensor(F.one_hot(labels, num_classes))
            y_0 = y_0.to(torch.float32).to(device)

            # # initialize loss
            L = loss_fn(y_0, y_0)
            L.zero_()
            
            pc = img_block_pos_expand(images, params['base_num'])
            pc = pc.to(torch.float32).to(device)
            x = point_net(pc)
            inf_x = g_inf_net(pc)
            # print(x.size())

            # take the sum from timestep T to 0, reverse order
            g_pixel_tensor = torch.flip(x, dims=[2])
            g_cumsum = torch.cumsum(g_pixel_tensor, dim=2).to(device)
            g_inf_tensor = torch.flip(inf_x, dims=[2])

            # train f_net
            f_out, L = f_net(T, g_cumsum, g_pixel_tensor, loss_fn, L, g_inf_tensor)

            # back propagation
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            print(f"Epoch [{e+1}/{num_epochs}], Loss: {L.item():.6f}")
            logging.info(f"Epoch [{e+1}/{num_epochs}], Loss: {L.item():.6f}")
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    return f_net, g_inf_net

if __name__ == "__main__":
    params = {
        # hyperparam
        "batch_size": 32,
        "num_classes": 10,
        "pixel_count": 28 * 28,
        "channel_count": 1,
        "T": 28 * 28,
        "lr_f": 0.001,
        "num_epochs": 5, 
        "noise_scale": 1e-3,
        "point_dimension": 11,
        "base_num": 2
    }

    fn_f = f"Mar30_f_rnn_v18_sp2_noTnet"
    fn_g_inf = f"Mar30_g_inf_sp2_noTnet"
    path_g_net = "Mar28_point_net_spatial2_noTnet.pth"

    # configurate logging function
    logging.basicConfig(filename = fn_f + ".log",
                        level = logging.DEBUG,
                        format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    # load the convolution part of pre-trained encoder
    g_trained_state_dict = torch.load(path_g_net)
    state_dict = {k: v for k, v in g_trained_state_dict.items() if 'base_pointnet' in k}  # Filter to get only 'i2h' parameters
    state_dict = {key.replace('base_pointnet.', ''): value for key, value in state_dict.items()}
    g_net = BasePointNet(point_dimension=params['point_dimension'])
    g_net.load_state_dict(state_dict)
    g_net = g_net.to(device)
    g_net.eval()
    PATH_f = fn_f + ".pth"
    PATH_g_inf = fn_g_inf + ".pth"
    # save model
    f_net, g_inf_net = train_f(g_net, params)
    torch.save(f_net.state_dict(), PATH_f)
    torch.save(g_inf_net.state_dict(), PATH_g_inf)