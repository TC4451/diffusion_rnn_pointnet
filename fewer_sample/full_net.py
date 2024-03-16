import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
import torch.nn.functional as F
from Point_Net import BasePointNet, ClassificationPointNet
from DiffRnn_Net import DRNet, DRNetTest

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.where(x > 0, 1, 0))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST datasets
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
portion_train_size = len(mnist_train_full) // 2
indices_train = torch.arange(portion_train_size)
mnist_train_half = Subset(mnist_train_full, indices_train)
trainloader = DataLoader(mnist_train_half, batch_size=64, shuffle=False)
testloader = DataLoader(mnist_test_full, batch_size=64, shuffle=False)


################################ Helper ################################
# region

def get_coords():
    # get coordinates of pixels
    coords_x, coords_y = torch.meshgrid(torch.arange(0, 28), torch.arange(0, 28))
    coords_x = coords_x.flatten().float().unsqueeze(1)
    coords_y = coords_y.flatten().float().unsqueeze(1)
    return coords_x, coords_y

# tranform image to 3D (x, y, binary value)
def img_to_3d(img, coords_x, coords_y):
    values = img.view(-1).unsqueeze(1)
    pc = torch.cat((coords_x, coords_y, values), dim=1)
    return pc

# endregion

################################ Train ################################

# region



# training of g net
def g_net_training(params, coords_x, coords_y):
    model = ClassificationPointNet(num_classes=10, point_dimension=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # training model
    epochs=params['num_epochs_g']
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    fn_g = params['fn_g']

    # get coordinates of pixels
    coords_x, coords_y = torch.meshgrid(torch.arange(0, 28), torch.arange(0, 28))
    coords_x = coords_x.flatten().float().unsqueeze(1)
    coords_y = coords_y.flatten().float().unsqueeze(1)

    logging.basicConfig(filename = fn_g + ".log",
                        level = logging.DEBUG,
                        format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    train_output = []
    train_ft = []
    test_output = []
    test_ft = []
    for epoch in range(epochs):
        epoch_train_loss = []
        epoch_train_acc = []

        # training loop
        for i, (images, labels) in enumerate(trainloader):
            batch_pc = []
            for img in images:
                batch_pc.append(img_to_3d(img, coords_x, coords_y))
            pc = torch.stack(batch_pc, dim=0)
            pc = pc.to(torch.float32).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            model = model.train()
            preds, feature_transform, tnet_out = model(pc)

            if epoch == epochs-1:
                train_output.append(preds)
                train_ft.append(feature_transform)

            identity = torch.eye(feature_transform.shape[-1])
            identity = identity.to(device)
            regularization_loss = torch.norm(
                identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
            # Loss
            loss = F.nll_loss(preds, labels) + 0.001 * regularization_loss
            epoch_train_loss.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
            preds = preds.data.max(1)[1]
            corrects = preds.eq(labels.data).cpu().sum()

            accuracy = corrects.item() / float(images.size(0))
            epoch_train_acc.append(accuracy)

        epoch_test_loss = []
        epoch_test_acc = []

        # validation loop
        for i, (val_images, val_labels) in enumerate(testloader):
            val_batch_pc = []
            for img in val_images:
                val_batch_pc.append(img_to_3d(img, coords_x, coords_y))
            val_pc = torch.stack(val_batch_pc, dim=0)
            val_pc = val_pc.to(torch.float32).to(device)
            val_labels = val_labels.to(device)
            model = model.eval()
            val_preds, feature_transform, tnet_out = model(val_pc)

            if epoch == epochs-1:
                test_output.append(val_preds)
                test_ft.append(feature_transform)

            val_loss = F.nll_loss(val_preds, val_labels)
            epoch_test_loss.append(val_loss.cpu().item())
            val_preds = val_preds.data.max(1)[1]
            corrects = val_preds.eq(val_labels.data).cpu().sum()
            accuracy = corrects.item() / float(val_images.size(0))
            epoch_test_acc.append(accuracy)

        print('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f'
                % (epoch,
                    round(np.mean(epoch_train_loss), 4),
                    round(np.mean(epoch_test_loss), 4),
                    round(np.mean(epoch_train_acc), 4),
                    round(np.mean(epoch_test_acc), 4)))

        logging.info('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f'
                % (epoch,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_test_loss), 4),
                round(np.mean(epoch_train_acc), 4),
                round(np.mean(epoch_test_acc), 4)))

        train_loss.append(np.mean(epoch_train_loss))
        test_loss.append(np.mean(epoch_test_loss))
        train_acc.append(np.mean(epoch_train_acc))
        test_acc.append(np.mean(epoch_test_acc))

    path_point_net = fn_g + ".pth"
    torch.save(model.state_dict(), path_point_net)

    return train_output, test_output, train_ft, test_ft

def f_net_training(g_output, params):
    fn_f = params['fn_f']
    num_classes = params['num_classes']
    lr = params['lr_f']
    num_epochs = params['num_epochs_f']
    T = params['T']
    # Initialize network
    f_net = DRNet().to(device)
    optimizer = torch.optim.AdamW(f_net.parameters(), lr)
    loss_fn = torch.nn.MSELoss().to(device)
    # f_outputs = []
    for e in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            # initialize y_0
            y_0 = torch.tensor(F.one_hot(labels, num_classes))
            y_0 = y_0.to(torch.float32).to(device)

            # # initialize loss
            L = loss_fn(y_0, y_0)
            L.zero_()
            
            x = g_output[i]

            # take the sum from timestep T to 0, reverse order
            g_pixel_tensor = torch.flip(x, dims=[2])
            g_cumsum = torch.cumsum(g_pixel_tensor, dim=2).to(device)

            # train f_net
            f_out, L = f_net(T, g_cumsum, g_pixel_tensor, loss_fn, L)
            # if e == num_epochs-1:
            #     f_outputs.append(f_out)

            # back propagation
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            print(f"Epoch [{e+1}/{num_epochs}], Loss: {L.item():.6f}")
            logging.info(f"Epoch [{e+1}/{num_epochs}], Loss: {L.item():.6f}")

    torch.save(f_net.state_dict(), fn_f + ".pth")
    return f_net.state_dict()

def d_net_training(f_net_sd, g_train_output, g_test_output, g_train_ft, g_test_ft, params):
    fn_d = params['fn_d']
    trained_f_net = DRNetTest().to(device)
    trained_f_net.load_state_dict(f_net_sd)
    trained_f_net.eval()
    encoding = []
    label_list = []
    feature_transform_list = []
    num_classes = params['num_classes']
    T = params['T']
    lr = params['lr_d']
    epochs = params['num_epochs_d']

    test_f_outputs = []

    # Sampling
    with torch.no_grad():
        # run f through train dataset
        for i, (images, labels) in enumerate(trainloader):
            x = g_train_output[i]
            feature_transform = g_train_ft[i]

            f_out = trained_f_net(T, x)
            
            # record the labels and y_0 predicted
            nlabels = labels.clone().detach()
            label_list.append(nlabels)
            ny_0 = f_out.clone().detach()
            encoding.append(ny_0)
            ft = feature_transform.clone().detach()
            feature_transform_list.append(ft)

        # run f through test dataset
        for i, (images, labels) in enumerate(testloader):
            x = g_test_output[i]
            feature_transform = g_test_ft[i]

            f_out = trained_f_net(T, x)
            test_f_outputs.append(f_out)
            
            # record the labels and y_0 predicted
            nlabels = labels.clone().detach()
            label_list.append(nlabels)
            ny_0 = f_out.clone().detach()
            encoding.append(ny_0)
            ft = feature_transform.clone().detach()
            feature_transform_list.append(ft)
    
    decoder = train_decoder(encoding, label_list, feature_transform_list, lr, epochs)
    torch.save(decoder.state_dict(), fn_d+".pth")
    return decoder, test_f_outputs

# training the decoder with the output from f net
def train_decoder(y_0_list, label_list, feature_transform_list, lr, epochs):
    # Initialize model
    model = ClassificationPointNet(num_classes=10, point_dimension=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for e in range(epochs):
        epoch_train_loss = []
        epoch_train_acc = []
        for i, (y_0, labels, ft) in enumerate(zip(y_0_list, label_list, feature_transform_list)):
            optimizer.zero_grad()
            labels = labels.to(device)
            # Train decoder
            model = model.train()
            # Make prediction
            preds = model(y_0)

            identity = torch.eye(ft.shape[-1])
            identity = identity.to(device)
            regularization_loss = torch.norm(
                identity - torch.bmm(ft, ft.transpose(2, 1)))
            # Loss
            loss = F.nll_loss(preds, labels) + 0.001 * regularization_loss
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



# endregion



################################ Test #################################

# region



def test_f(decoder, test_f_outputs, g_test_output):
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
            x = g_test_output[i]
            f_out = test_f_outputs[i]

            # Use pre-trained decoder classification
            y_pred = decoder(f_out)
            _, predicted = torch.max(y_pred.data, 1)
            labels = labels.to(device)
            accuracy = (predicted == labels).sum().item() / predicted.size(0)
            accuracy_list.append(accuracy)
            print('test accuracy: {}'.format(accuracy))

            
            if i%2==0:
                encoding.append(f_out)
                original_pixel.append(x.view(images.size(0), -1))
                labels_list.append(labels.to(device))
            # predictions.append(y_pred)
    
    print("avg accuracy: {}".format(sum(accuracy_list) / len(accuracy_list)))

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
    plt.savefig('Mar15_tsne_y0_noise_scale_e300_neg1.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for label in range(num_classes):
        idx = labels_array == label
        plt.scatter(tsne_pixel[idx, 0], tsne_pixel[idx, 1], s=10, color=distinct_colors[label], label=str(label))
    plt.title('t-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('Mar15_tsne_g_pixel_noise_scale_e300_neg1.png')


# endregion


if __name__ == "__main__":
    params = {
        # hyperparam
        "batch_size": 64,
        "num_classes": 10,
        "pixel_count": 28 * 28,
        "channel_count": 1,
        "T": 28 * 28,
        "lr_f": 0.001,
        "num_epochs_g": 80, 
        "num_epochs_f": 10,
        "noise_scale": 1e-3,
        "point_dimension": 3,
        "train_data_portion": "half",
        'lr_d':0.001,
        'num_epochs_d':300
    }
    params['fn_g'] = f"Mar16_point_net_v3_{params['train_data_portion']}"
    params['fn_f'] = f"Mar16_f_rnn_v18_{params['train_data_portion']}_lr{params['lr_f']}_e{params['num_epochs_f']}"
    params['fn_d'] = f"Mar16_decoder_{params['train_data_portion']}"

    coords_x, coords_y = get_coords()
    g_train_output, g_test_output, g_train_ft, g_test_ft = g_net_training(params, coords_x, coords_y)
    f_net_sd = f_net_training(g_train_output, params)
    d_net, test_f_outputs = d_net_training(f_net_sd, g_train_output, g_test_output, g_train_ft, g_test_ft, params)
    test_f(d_net, test_f_outputs, g_test_output)


    


    
