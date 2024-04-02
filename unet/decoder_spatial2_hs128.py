# https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import math
import logging
from unet_parts import *


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
device = torch.device("cuda")

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 128)
        self.down1 = Down(128, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(192, 128, bilinear)
        # self.outc = OutConv(64, n_classes)
        # self.fc = nn.Sequential(
        #     nn.Linear(64, 10),
        #     nn.BatchNorm1d(10), 
        #     nn.ReLU()
        # )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # logits = self.outc(x)
        # x = self.outc(x)
        x = x.view(x.size(0), x.size(1), -1)
        # x = torch.sum(x, dim=2).view(x.size(0), -1)
        # x = self.fc(x)
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
        self.conv_5 = nn.Conv1d(128, 128, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(128)
        

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
        # x, ix = nn.MaxPool1d(num_points, return_indices=True)(x)  # max-pooling
        # x = x.view(-1, 256)  # global feature vector 

        return x, feature_transform, tnet_out
    
class DRNetTest(nn.Module):
    def __init__(self):
        super(DRNetTest, self).__init__()

        self.h2h = nn.Sequential(
            nn.Linear(128, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            nn.Linear(784, 128),
        )

    def forward(self, T, g_pixel_tensor, inf_x):
        latent_y_T = torch.zeros(g_pixel_tensor.size(0), 128).to(device)

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

# # model
# class ClassificationPointNet(nn.Module):

#     def __init__(self, num_classes, dropout=0.3, point_dimension=3):
#         super(ClassificationPointNet, self).__init__()

#         self.fc_1 = nn.Linear(256, 128)
#         self.fc_2 = nn.Linear(128, 64)
#         self.fc_3 = nn.Linear(64, num_classes)

#         self.bn_1 = nn.BatchNorm1d(128)
#         self.bn_2 = nn.BatchNorm1d(64)

#         self.dropout_1 = nn.Dropout(dropout)

#     def forward(self, x):
#         x = x.view(-1, 256)

#         x = F.relu(self.bn_1(self.fc_1(x)))
#         x = F.relu(self.bn_2(self.fc_2(x)))
#         x = self.dropout_1(x)

#         return F.log_softmax(self.fc_3(x), dim=1)

class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # print(x)
        # print(f"x size: {x.size()}")
        # x = torch.sum(x, dim=2).view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# -----------

# run f net and get the result before training the decoder
def train_d(trained_f_net, trained_g_inf_net, trained_g_net, params):
    encoding = []
    label_list = []
    num_classes = params['num_classes']
    T = params['T']
    lr = params['lr_d']
    epochs = params['num_epochs_d']

    # Sampling
    with torch.no_grad():
        # run f through train dataset
        for i, (images, labels) in enumerate(trainloader):
            pc = img_block_pos_expand(images, params['base_num'])
            pc = pc.to(torch.float32).to(device)
            t_pc = pc.view(pc.size(0), params['point_dimension'], 28, 28)
            x = trained_g_net(t_pc)
            inf_x, inf_ft, inf_tnet_out = trained_g_inf_net(pc)

            f_out = trained_f_net(T, x, inf_x)
            
            # record the labels and y_0 predicted
            nlabels = labels.clone().detach()
            label_list.append(nlabels)
            ny_0 = f_out.clone().detach()
            encoding.append(ny_0)

        # run f through test dataset
        for i, (images, labels) in enumerate(testloader):
            pc = img_block_pos_expand(images, params['base_num'])
            pc = pc.to(torch.float32).to(device)
            t_pc = pc.view(pc.size(0), params['point_dimension'], 28, 28)
            x = trained_g_net(t_pc)
            inf_x, inf_ft, inf_tnet_out = trained_g_inf_net(pc)

            f_out = trained_f_net(T, x, inf_x)
            
            # record the labels and y_0 predicted
            nlabels = labels.clone().detach()
            label_list.append(nlabels)
            ny_0 = f_out.clone().detach()
            encoding.append(ny_0)

    y_0_data_path = "y_0_encoding_hs128.pt"
    torch.save(encoding, y_0_data_path)
    label_data_path = "label_list_hs128.pt"
    torch.save(label_list, label_data_path)
    
    decoder = train_decoder(encoding, label_list, lr, epochs)

    return decoder

# training the decoder with the output from f net
def train_decoder(y_0_list, label_list, lr, epochs):
    # Initialize model
    model = ClassificationNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for e in range(epochs):
        epoch_train_loss = []
        epoch_train_acc = []
        for i, (y_0, labels) in enumerate(zip(y_0_list, label_list)):
            optimizer.zero_grad()
            labels = labels.to(device)
            # Train decoder
            model = model.train()
            # Make prediction
            preds = model(y_0)

            # Loss
            loss = F.nll_loss(preds, labels)
            # Back propagations
            epoch_train_loss.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
            preds = preds.data.max(1)[1]
            corrects = preds.eq(labels.data).cpu().sum()

            accuracy = corrects.item() / float(y_0.size(0))
            epoch_train_acc.append(accuracy)
        
        print('Epoch %s: train loss: %s, train accuracy: %s'
              % (e,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_train_acc), 4)))

        logging.info('Epoch %s: train loss: %s, train accuracy: %s'
              % (e,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_train_acc), 4)))
    return model


if __name__ == "__main__":
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

    fn_d = f"Apr1_decoder_hs128"
    # configurate logging function
    logging.basicConfig(filename = fn_d + ".log",
                        level = logging.DEBUG,
                        format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    # path to saved models
    PATH_g = "Mar30_unet_hs128.pth"
    PATH_f = f"Mar31_f_rnn_v21_hs128.pth"
    PATH_g_inf = f"Mar31_g_inf_hs128.pth"
    PATH_d = fn_d + ".pth"

    # load g net
    g_trained_state_dict = torch.load(PATH_g)
    state_dict = {k: v for k, v in g_trained_state_dict.items() if 'fc' not in k}  # Filter to get only 'i2h' parameters
    trained_g_net = UNet(n_channels=11, n_classes=10)
    trained_g_net.load_state_dict(state_dict)
    trained_g_net = trained_g_net.to(device)
    trained_g_net.eval()
    # load f net
    trained_f_net = DRNetTest().to(device)
    trained_f_net.load_state_dict(torch.load(PATH_f, map_location=torch.device('cpu')), strict=False)
    trained_f_net.eval()
    # load g inf net
    trained_g_inf_net = BasePointNet(point_dimension=params['point_dimension']).to(device)
    trained_g_inf_net.load_state_dict(torch.load(PATH_g_inf, map_location=torch.device('cpu')), strict=False)
    trained_g_inf_net.eval()
    # save model
    decoder = train_d(trained_f_net, trained_g_inf_net, trained_g_net, params)
    # y_0_data_path = "y_0_encoding.pt"
    # y_0_list = torch.load(y_0_data_path)
    # label_data_path = "laebl_list.pt"
    # label_list = torch.load(label_data_path)
    # decoder = train_decoder(y_0_list, label_list, params['lr_d'], params['num_epochs_d'])
    torch.save(decoder.state_dict(), PATH_d)
    
    # # test model
    # trained_decoder = Decoder().to(device)
    # trained_decoder.load_state_dict(torch.load(PATH_d, map_location=torch.device('cpu')), strict=False)
    # trained_decoder.eval()
    # test_f(trained_f_net, trained_decoder, params)