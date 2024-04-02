# CUDA_LAUNCH_BLOCKING=1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import math
import matplotlib as plt
import logging
import numpy as np

# load MNIST dataset, convert to binary pixel values
mnist_train = datasets.MNIST(root='/mnt/VOL1/fangzhou/local/data/zilin_data/data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: torch.where(x > 0,1,0))
                             ]))
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=False)
mnist_test = datasets.MNIST(root='/mnt/VOL1/fangzhou/local/data/zilin_data/data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: torch.where(x > 0,1,0))
                             ]))
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
# set device to run
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
#         # if torch.cuda.is_available():
#         identity_matrix = identity_matrix.cuda()
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

    def forward(self, T, g_pixel_tensor, inf_x):
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
            learned_x_point = inf_x[:, :, t_prime].view(g_pixel_tensor.size(0), -1)
            # predict the link to embedding of next timestep
            input = latent_y_T + learned_x_point
            out = self.h2h(input)
            # calculate the location of embedding of next timestep
            latent_y_T = latent_y_T + x_point + out + sigma_t*epsilon

        return latent_y_T

# model
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

# Test function for f
def test_f(point_net, trained_f_net, trained_g_inf_net, trained_decoder, params):
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
            pc = img_block_pos_expand(images, params['base_num'])
            pc = pc.to(torch.float32).to(device)
            x  = point_net(pc)
            inf_x = trained_g_inf_net(pc)

            f_out = trained_f_net(T, x, inf_x)

            # Use pre-trained decoder classification
            y_pred = trained_decoder(f_out)

            preds = y_pred.data.max(1)[1]
            labels = labels.to(device)
            corrects = preds.eq(labels.data).cpu().sum()

            accuracy = corrects.item() / float(y_pred.size(0))
            # _, predicted = torch.max(y_pred.data, 1)
            # labels = labels.to(device)
            # accuracy = (predicted == labels).sum().item() / predicted.size(0)
            accuracy_list.append(accuracy)
            print('test accuracy: {}'.format(accuracy))

            
            if i%2==0:
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
    plt.title('t-SNE Projection of f-net Output')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Mar29_tsne_y0.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for label in range(num_classes):
        idx = labels_array == label
        plt.scatter(tsne_pixel[idx, 0], tsne_pixel[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    plt.title('t-SNE Projection of g-net Output')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Mar29_tsne_g_pixel.png')


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
        "batch_size": 64,
        "num_classes": 10,
        "pixel_count": 28 * 28,
        "channel_count": 1,
        "T": 28 * 28,
        "lr_f": 0.001,
        "num_epochs_f": 10, 
        "num_epochs_d": 200, 
        "noise_scale": 1e-3,
        "lr_d": 0.001,
        "point_dimension": 11,
        "base_num": 2
    }

    # # configurate logging function
    # logging.basicConfig(filename = f"Mar8_f_rnn_v18_AvgPool_lr{params['lr_f']}_e{params['num_epochs']}_loss.log",
    #                     level = logging.DEBUG,
    #                     format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    # load the convolution part of pre-trained encoder
    path_g_net = "Mar28_point_net_spatial2_noTnet.pth"
    g_trained_state_dict = torch.load(path_g_net)
    state_dict = {k: v for k, v in g_trained_state_dict.items() if 'base_pointnet' in k}  # Filter to get only 'i2h' parameters
    state_dict = {key.replace('base_pointnet.', ''): value for key, value in state_dict.items()}
    g_net = BasePointNet(point_dimension=params['point_dimension'])
    g_net.load_state_dict(state_dict)
    g_net = g_net.to(device)
    g_net.eval()
    PATH_f = f"Mar30_f_rnn_v18_sp2_noTnet.pth"
    PATH_g_inf = f"Mar30_g_inf_sp2_noTnet.pth"
    
    # load model for testing
    trained_f_net = DRNetTest().to(device)
    trained_f_net.load_state_dict(torch.load(PATH_f, map_location=torch.device('cpu')), strict=False)
    trained_f_net.eval()
    # load g inf net
    trained_g_inf_net = BasePointNet(point_dimension=params['point_dimension']).to(device)
    trained_g_inf_net.load_state_dict(torch.load(PATH_g_inf, map_location=torch.device('cpu')), strict=False)
    trained_g_inf_net.eval()

    # load decoder for testing
    PATH_d = 'Apr1_decoder_sp2_noTnet.pth'
    trained_decoder = ClassificationPointNet(num_classes=10, point_dimension=3).to(device)
    trained_decoder.load_state_dict(torch.load(PATH_d, map_location=torch.device('cpu')), strict=False)
    trained_decoder.eval()
    test_f(g_net, trained_f_net, trained_g_inf_net, trained_decoder, params)