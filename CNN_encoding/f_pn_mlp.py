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

# Convolution part of the g_net, pretrained in g_linear_net.py 
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
        return x

# Diffusion rnn network
class DRNet(nn.Module):
    def __init__(self):
        super(DRNet, self).__init__()
        
        # hidden layer to previous timestep
        self.h2h = nn.Sequential(
            nn.Linear(512, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 512),
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
            epsilon = 1e-2 * torch.randn_like(latent_y_0)
            
            # locate the current y_t by cumulated sum of g_net output with added noise
            y_t = latent_y_0 - a_t/a_T * g_cumsum[:, :, t_prime-1] + sigma_t*epsilon

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

# Testing network
class DRNetTest(nn.Module):
    def __init__(self):
        super(DRNetTest, self).__init__()

        # hidden layer to previous timestep
        self.h2h = nn.Sequential(
            nn.Linear(512, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 512),
        )

    def forward(self, T, g_pixel_tensor):
        latent_y_T = torch.zeros(g_pixel_tensor.size(0), 512).to(device)
        
        # t ~ Uniform ({1, ...T})
        timestep = torch.arange(T, 0, -1)

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

# Decoder network, pretrained in d_decoder_mlp.py
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

# Train function for f
def train_f(g_net,  g_fc_weight, g_fc_bias, PATH_f, params):
    num_classes = params['num_classes']
    lr = params['lr_f']
    num_epochs = params['num_epochs']
    T = params['T']
    # Initialize network
    f_net = DRNet().to(device)
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
            
            # flatten the pixels as sequential input
            flat_image = images.view(images.size(0), 1, 1, -1)
            # Encoder for image
            g_out = g_net(flat_image).to(device)    #(64, 32*784)
            # Get the encoded pixels independently using pre_trained weights from g, 32 refers to the features for each pixel
            g_pixel_list = []
            for t in range(0, g_out.size(1), 32):
                g_o = g_out[:, t:t+32]
                g_w = g_fc_weight[t:t+32, :]
                g_t = torch.matmul(g_o, g_w) + g_fc_bias
                g_pixel_list.append(g_t)
                t = t+32
            g_pixel_tensor = torch.stack(g_pixel_list, dim=2)

            # take the sum from timestep T to 0, reverse order
            g_pixel_tensor = torch.flip(g_pixel_tensor, dims=[2])
            g_cumsum = torch.cumsum(g_pixel_tensor, dim=2).to(device)

            # train f_net
            f_out, L = f_net(T, g_cumsum, g_pixel_tensor, loss_fn, L)

            # back propagation
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            print(f"Epoch [{e+1}/{num_epochs}], Loss: {L.item():.6f}")
            logging.info(f"Epoch [{e+1}/{num_epochs}], Loss: {L.item():.6f}")
        
    return f_net

# Test function for f
def test_f(trained_f_net, trained_decoder, params):
    encoding = []
    predictions = []
    original_pixel = []
    accuracy_list = []
    labels_list = []
    num_classes = params['num_classes']
    T = params['T']

    # Sampling
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
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
            # Run through diffusion rnn net to get y_0
            f_out = trained_f_net(T, g_pixel_tensor)
            # Use pre-trained decoder classification
            y_pred = trained_decoder(f_out)
            _, predicted = torch.max(y_pred.data, 1)
            labels = labels.to(device)
            accuracy = (predicted == labels).sum().item() / predicted.size(0)
            accuracy_list.append(accuracy)
            print('test accuracy: {}'.format(accuracy))

            encoding.append(f_out)
            original_pixel.append(g_pixel_tensor.view(images.size(0), -1))
            labels_list.append(labels.to(device))
            predictions.append(y_pred)
    
    print("avg accuracy: {}".format(sum(accuracy_list) / len(accuracy_list)))

    cat_encoding = torch.cat(encoding, dim=0).to(device)
    all_labels = torch.cat(labels_list, dim=0).to(device)
    original_pixel = torch.cat(original_pixel, dim=0).to(device)
    predictions = torch.cat(predictions, dim=0).to(device)

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Convert tensor to NumPy array
    data_array = cat_encoding.cpu().detach().numpy()
    labels_array = all_labels.cpu().detach().numpy()
    original_pixel = original_pixel.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()

    # Perform t-SNE embedding
    tsne = TSNE(n_components=2)
    tsne_data = tsne.fit_transform(data_array)
    tsne2 = TSNE(n_components=2)
    tsne_pixel = tsne2.fit_transform(original_pixel)
    tsne3 = TSNE(n_components=2)
    tsne_pred = tsne3.fit_transform(predictions)
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Plot t-SNE embedding
    plt.figure(figsize=(8, 6))
    for label in range(num_classes):
        idx = labels_array == label
        plt.scatter(tsne_data[idx, 0], tsne_data[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    plt.title('t-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Feb23_tsne_y0.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for label in range(num_classes):
        idx = labels_array == label
        plt.scatter(tsne_pixel[idx, 0], tsne_pixel[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    plt.title('t-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Feb23_tsne_g_pixel.png')

    plt.figure(figsize=(8, 6))
    for label in range(num_classes):
        idx = labels_array == label
        plt.scatter(tsne_pred[idx, 0], tsne_pred[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    plt.title('t-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Feb23_tsne_ypred.png')

    

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    params = {
        # hyperparam
        "batch_size": 64,
        "num_classes": 10,
        "pixel_count": 28 * 28,
        "channel_count": 1,
        "T": 28 * 28,
        "lr_f": 0.001,
        "num_epochs": 10, 
        "noise_scale": 1e-3,
    }

    # configurate logging function
    logging.basicConfig(filename = f"Feb23_f_rnn_noi2l_lr{params['lr_f']}_e{params['num_epochs']}_loss.log",
                        level = logging.DEBUG,
                        format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    # load the convolution part of pre-trained encoder
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
    PATH_f = f"Feb23_f_rnn_noi2l_lr{params['lr_f']}_e{params['num_epochs']}.pth"
    # save model
    # f_net = train_f(g_net, g_fc_weight, g_fc_bias, PATH_f, params)
    # torch.save(f_net.state_dict(), PATH_f)

    # load model for testing
    trained_f_net = DRNetTest().to(device)
    trained_f_net.load_state_dict(torch.load(PATH_f, map_location=torch.device('cpu')), strict=False)
    trained_f_net.eval()
    # load decoder for testing
    PATH_d = 'Feb23_d_mlp_lr0.001_e10.pth'
    trained_decoder = Decoder().to(device)
    trained_decoder.load_state_dict(torch.load(PATH_d, map_location=torch.device('cpu')), strict=False)
    trained_decoder.eval()
    test_f(trained_f_net, trained_decoder, params)





