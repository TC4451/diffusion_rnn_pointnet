import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import math
import matplotlib as plt
import logging
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])
# download and load data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=16)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=16)

# set device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=3, stride = 1, padding=1),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=3, stride = 1, padding=1),                
        )

        for param in self.parameters():
            param.requires_grad = False

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2
        x = x.view(x.size(0), -1)   
        # x = self.fc1(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.fc(x)
        out = self.softmax(x)
        return out
    
class DRNetTest(nn.Module):
    def __init__(self):
        super(DRNetTest, self).__init__()

        self.i2l = nn.Sequential(
            nn.Linear(10, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.h2h = nn.Sequential(
            nn.Linear(512, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 512),
        )

    def forward(self, y_T, T, g_pixel_tensor):
        latent_y_T = self.i2l(y_T)

        # t ~ Uniform ({1, ...T})
        timestep = torch.arange(T, 0, -1)
        # timestep = torch.arange(1, T+1, 1)

        for t in timestep:
            # set eq
            a_t = (t+1)/(T+1)
            t_prime = T-t 
            sigma_t = math.sqrt(a_t*(1-a_t))

            # epsilon ~ N(0, I) * 1e-2
            epsilon = 1e-2 * torch.randn_like(latent_y_T)
            
            # locate the g(x) at current timepoint
            x_point = g_pixel_tensor[:, :, t_prime].view(g_pixel_tensor.size(0), -1)
            # predict the link to embedding of next timestep
            input = latent_y_T + x_point + sigma_t*epsilon
            out = self.h2h(input)
            # calculate the location of embedding of next timestep
            latent_y_T = latent_y_T + x_point + out

        return latent_y_T
    
# -----------

# run f net and get the result before training the decoder
def train_d(trained_f_net, g_net,  g_fc_weight, g_fc_bias, params):
    train_encoding = []
    train_label_list = []
    test_encoding = []
    test_label_list = []
    num_classes = params['num_classes']
    T = params['T']
    lr = params['lr_d']
    epochs = params['num_epochs_d']

    # Sampling
    with torch.no_grad():
        # run f through train dataset
        for i, (images, labels) in enumerate(trainloader):
            # initialize y_T
            y_T = torch.zeros(images.size(0), num_classes).to(device)

            # (64, 784)
            flat_image = images.view(images.size(0), 1, 1, -1)
            # Encoder for image
            g_out = g_net(flat_image).to(device)    #(64, 32*784)
            g_pixel_list = []
            for t in range(0, g_out.size(1), 32):
                g_o = g_out[:, t:t+32]
                g_w = g_fc_weight[t:t+32, :]
                g_t = torch.matmul(g_o, g_w) + g_fc_bias
                g_pixel_list.append(g_t)
                t = t+32

            g_pixel_tensor = torch.stack(g_pixel_list, dim=2)
            f_out = trained_f_net(y_T, T, g_pixel_tensor)
            
            # record the labels and y_0 predicted
            nlabels = labels.clone().detach()
            train_label_list.append(nlabels)
            ny_0 = f_out.clone().detach()
            train_encoding.append(ny_0)

        # run f through test dataset
        for i, (images, labels) in enumerate(testloader):
            # initialize y_T
            y_T = torch.randn(images.size(0), num_classes).to(device)

            # (64, 784)
            flat_image = images.view(images.size(0), 1, 1, -1)
            # Encoder for image
            g_out = g_net(flat_image).to(device)    #(64, 32*784)
            g_pixel_list = []
            for t in range(0, g_out.size(1), 32):
                g_o = g_out[:, t:t+32]
                g_w = g_fc_weight[t:t+32, :]
                g_t = torch.matmul(g_o, g_w) + g_fc_bias
                g_pixel_list.append(g_t)
                t = t+32

            g_pixel_tensor = torch.stack(g_pixel_list, dim=2)
            f_out = trained_f_net(y_T, T, g_pixel_tensor)
            
            # record the labels and y_0 predicted
            nlabels = labels.clone().detach()
            test_label_list.append(nlabels)
            ny_0 = f_out.clone().detach()
            test_encoding.append(ny_0)
    
    cat_train_encoding = torch.cat(train_encoding, dim=0).to(device)
    cat_train_label = torch.cat(train_label_list, dim=0).to(device)
    cat_test_encoding = torch.cat(test_encoding, dim=0).to(device)
    cat_test_label = torch.cat(test_label_list, dim=0).to(device)

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Convert tensor to NumPy array
    train_data_array = cat_train_encoding.cpu().detach().numpy()
    test_data_array = cat_test_encoding.cpu().detach().numpy()
    train_label_array = cat_train_label.cpu().detach().numpy()
    test_label_array = cat_test_label.cpu().detach().numpy()

    # Perform t-SNE embedding
    tsne = TSNE(n_components=2)
    tsne_train_data = tsne.fit_transform(train_data_array)
    tsne2 = TSNE(n_components=2)
    tsne_test_data = tsne2.fit_transform(test_data_array)
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Plot t-SNE embedding
    plt.figure(figsize=(8, 6))
    for label in range(num_classes):
        idx = train_label_array == label
        plt.scatter(tsne_train_data[idx, 0], tsne_train_data[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    plt.title('t-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Feb23_tsne_train_y0.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for label in range(num_classes):
        idx = test_label_array == label
        plt.scatter(tsne_test_data[idx, 0], tsne_test_data[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    plt.title('t-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Feb23_tsne_test_y0.png')


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    params = {
        # hyperparam
        "batch_size": 64,
        "num_classes": 10,
        "pixel_count": 28 * 28,
        "channel_count": 1,
        "T": 28 * 28,
        "lr_f": 0.001,
        "num_epochs_f": 10, 
        "num_epochs_d": 10, 
        "noise_scale": 1e-3,
        "lr_d": 0.001
    }

    # load the convolution part of pre-trained encoder (g)
    path_g_net = "Feb23_g_linear_lr0.005_e10.pth"
    g_trained_net = torch.load(path_g_net)
    state_dict = {k: v for k, v in g_trained_net.items() if 'conv' in k}  # Filter to get only 'i2h' parameters
    g_net = Encoder()
    g_net.load_state_dict(state_dict)
    g_net.eval()
    # load the fc part of pre-trained encoder (g) to hidden dimension
    state_dict_fc = {k: v for k, v in g_trained_net.items() if 'fc1' in k}
    g_fc_weight = torch.transpose(state_dict_fc['fc1.0.weight'], 0, 1) #(32*784, 512)
    g_fc_bias = state_dict_fc['fc1.0.bias'] #(512)
    PATH_f = f"Feb23_f_rnn_noi2l_lr{params['lr_f']}_e{params['num_epochs_f']}.pth"
    PATH_d = f"Feb23_d_mlp_lr{params['lr_d']}_e{params['num_epochs_d']}.pth"

    # load f net
    trained_f_net = DRNetTest().to(device)
    trained_f_net.load_state_dict(torch.load(PATH_f, map_location=torch.device('cpu')), strict=False)
    trained_f_net.eval()
    # save model
    decoder = train_d(trained_f_net, g_net,  g_fc_weight, g_fc_bias, params)
    
    # # test model
    # trained_decoder = Decoder().to(device)
    # trained_decoder.load_state_dict(torch.load(PATH_d, map_location=torch.device('cpu')), strict=False)
    # trained_decoder.eval()
    # test_f(trained_f_net, trained_decoder, params)