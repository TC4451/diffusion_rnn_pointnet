import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import math
import matplotlib as plt
import logging
import numpy as np

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.where(x > 0, 1, 0))
])

# Load the full MNIST datasets
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Calculate half the size of each dataset
half_train_size = len(mnist_train_full) // 2

# Create subsets for the first half of each dataset
indices_train = torch.arange(half_train_size)
mnist_train_half = Subset(mnist_train_full, indices_train)

# Create data loaders for the subsets
trainloader = DataLoader(mnist_train_half, batch_size=64, shuffle=False)
testloader = DataLoader(mnist_test_full, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# encoder model  
class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)

        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, self.output_dim*self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        # if torch.cuda.is_available():
        identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):

    def __init__(self, point_dimension):
        super(BasePointNet, self).__init__()
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)
        
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
        
        input_transform = self.input_transform(x) # T-Net tensor [batch, 3, 3]
        x = torch.bmm(x, input_transform) # Batch matrix-matrix product 
        x = x.transpose(2, 1) 
        tnet_out=x.cpu().detach().numpy()
        
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = x.transpose(2, 1)

        feature_transform = self.feature_transform(x) # T-Net tensor [batch, 64, 64]
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))

        return x, feature_transform, tnet_out

class ClassificationPointNet(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=3):
        super(ClassificationPointNet, self).__init__()

        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, num_classes)

        self.bn_1 = nn.BatchNorm1d(128)
        self.bn_2 = nn.BatchNorm1d(64)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, 256)

        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)

        return F.log_softmax(self.fc_3(x), dim=1)
    
# tranform image to 3D (x, y, binary value)
def img_to_3d(img):
    # get coordinates of pixels
    coords_x, coords_y = torch.meshgrid(torch.arange(0, img.size(1)), torch.arange(0, img.size(2)))
    coords_x = coords_x.flatten().float().unsqueeze(1)
    coords_y = coords_y.flatten().float().unsqueeze(1)
    values = img.view(-1).unsqueeze(1)
    pc = torch.cat((coords_x, coords_y, values), dim=1)
    return pc

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

# Test function for f
def test_f(point_net, trained_f_net, trained_decoder, params):
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
            batch_pc = []
            for img in images:
                batch_pc.append(img_to_3d(img))
            pc = torch.stack(batch_pc, dim=0)
            pc = pc.to(torch.float32).to(device)
            x, feature_transform, tnet_out = point_net(pc)
            f_out = trained_f_net(T, x)

            # Use pre-trained decoder classification
            y_pred = trained_decoder(f_out)
            _, predicted = torch.max(y_pred.data, 1)
            labels = labels.to(device)
            accuracy = (predicted == labels).sum().item() / predicted.size(0)
            accuracy_list.append(accuracy)
            print('test accuracy: {}'.format(accuracy))

            
            if i%3==0:
                encoding.append(f_out)
                original_pixel.append(x.view(images.size(0), -1))
                labels_list.append(labels.to(device))
            # predictions.append(y_pred)
    
    print("avg accuracy: {}".format(sum(accuracy_list) / len(accuracy_list)))

    import gc
    # import torch
    gc.collect()
    torch.cuda.empty_cache()

    cat_encoding = torch.cat(encoding, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    torch.cuda.empty_cache()
    original_pixel = torch.cat(original_pixel, dim=0)
    # predictions = torch.cat(predictions, dim=0)

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Convert tensor to NumPy array
    data_array = cat_encoding.cpu().detach().numpy()
    labels_array = all_labels.cpu().detach().numpy()
    original_pixel = original_pixel.cpu().detach().numpy()
    # predictions = predictions.cpu().detach().numpy()

    # Perform t-SNE embedding
    tsne = TSNE(n_components=2)
    tsne_data = tsne.fit_transform(data_array)
    tsne2 = TSNE(n_components=2)
    tsne_pixel = tsne2.fit_transform(original_pixel)
    # tsne3 = TSNE(n_components=2)
    # tsne_pred = tsne3.fit_transform(predictions)
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
    plt.savefig('Mar17_tsne_y0_50.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for label in range(num_classes):
        idx = labels_array == label
        plt.scatter(tsne_pixel[idx, 0], tsne_pixel[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    plt.title('t-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Mar17_tsne_g_pixel_50.png')

    # plt.figure(figsize=(8, 6))
    # for label in range(num_classes):
    #     idx = labels_array == label
    #     plt.scatter(tsne_pred[idx, 0], tsne_pred[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    # plt.title('t-SNE Plot')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.grid(True)
    # plt.savefig('Feb23_tsne_ypred.png')

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # export TORCH_CUDNN_V8_API_DISABLED=1
    params = {
        # hyperparam
        "batch_size": 32,
        "num_classes": 10,
        "pixel_count": 28 * 28,
        "channel_count": 1,
        "T": 28 * 28,
        "lr_f": 0.001,
        "num_epochs": 10, 
        "noise_scale": 1e-3,
        "point_dimension": 3
    }

    # # configurate logging function
    # logging.basicConfig(filename = f"Mar8_f_rnn_v18_AvgPool_lr{params['lr_f']}_e{params['num_epochs']}_loss.log",
    #                     level = logging.DEBUG,
    #                     format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    # load the convolution part of pre-trained encoder
    path_g_net = "Mar15_point_net_v3_50_train.pth"
    g_trained_state_dict = torch.load(path_g_net)
    state_dict = {k: v for k, v in g_trained_state_dict.items() if 'base_pointnet' in k}  # Filter to get only 'i2h' parameters
    state_dict = {key.replace('base_pointnet.', ''): value for key, value in state_dict.items()}
    g_net = BasePointNet(point_dimension=params['point_dimension'])
    g_net.load_state_dict(state_dict)
    g_net = g_net.to(device)
    g_net.eval()
    PATH_f = f"Mar16_f_rnn_v18_50.pth"
    
    # load model for testing
    trained_f_net = DRNetTest().to(device)
    trained_f_net.load_state_dict(torch.load(PATH_f, map_location=torch.device('cpu')), strict=False)
    trained_f_net.eval()

    # load decoder for testing
    PATH_d = 'Mar16_decoder_50.pth'
    trained_decoder = ClassificationPointNet(num_classes=10, point_dimension=3).to(device)
    trained_decoder.load_state_dict(torch.load(PATH_d, map_location=torch.device('cpu')), strict=False)
    trained_decoder.eval()
    test_f(g_net, trained_f_net, trained_decoder, params)