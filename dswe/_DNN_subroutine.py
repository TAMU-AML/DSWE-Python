# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def get_default_device():
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class NeuralNets(nn.Module):
    def __init__(self, n_feats=5, feats_scale=4):
        super(NeuralNets, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(n_feats, 8 * feats_scale),
            nn.ReLU(inplace=True),

            nn.Linear(8 * feats_scale, 16 * feats_scale),
            nn.ReLU(inplace=True),

            nn.Linear(16 * feats_scale, 32 * feats_scale),
            nn.ReLU(inplace=True),

            nn.Linear(32 * feats_scale, 64 * feats_scale),
            nn.ReLU(inplace=True),

            nn.Linear(64 * feats_scale, 32 * feats_scale),
            nn.ReLU(inplace=True),

            nn.Linear(32 * feats_scale, 16 * feats_scale),
            nn.ReLU(inplace=True),

            nn.Linear(16 * feats_scale, 1),
        )

    def forward(self, X):
        x = self.features(X)
        return x


def select_optimizer(name, params, lr, momentum):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)


def select_loss_function(name):
    if name == "l1_loss":
        return nn.L1Loss()
    if name == "l2_loss":
        return nn.MSELoss()


def train_model(n_epochs, model, train_dl, val_dl, batch_size, device, loss_fn, optimizer, lr, momentum, print_log=False):
    """Record these values the end of each epoch"""
    train_losses, val_losses = [], []
    criterion = select_loss_function(loss_fn)

    patience = 0
    min_loss = 1e5

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        optimizer_fn = select_optimizer(
            optimizer, model.parameters(), lr, momentum)
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = data.to(device), target.to(device)
            optimizer_fn.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_fn.step()
            train_loss += loss.item()

        train_losses.append(train_loss / (batch_size * len(train_dl)))

        model.eval()
        if len(val_dl) > 0:
            val_loss = 0
            for batch_idx, (data, target) in enumerate(val_dl):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

            if val_loss < min_loss:
                min_loss = val_loss
                patience = 0
            else:
                patience += 1
                lr = lr / 5

            val_losses.append(val_loss / (batch_size * len(val_dl)))
            if print_log:
                print("Epoch : {}, Train Loss : {:.4f}, Val Loss : {:.4f}".format(
                    epoch, train_loss / (batch_size * len(train_dl)), val_loss / (batch_size * len(val_dl))))

            if (patience > 3 or val_loss > 2 * min_loss) and epoch > 6:
                if print_log:
                    print("Overfitting started. Early stopping!!")
                break
        else:
            if train_loss < min_loss:
                min_loss = train_loss
                patience = 0
            else:
                patience += 1
                lr = lr / 5

            if print_log:
                print("Epoch : {}, Train Loss : {:.4f}".format(
                    epoch, train_loss / (batch_size * len(train_dl))))

            if (patience > 3 or train_loss > 2 * min_loss) and epoch > 6:
                if print_log:
                    print("Overfitting started. Early stopping!!")
                break

    return model, train_losses, val_losses


def plot_losses(train_losses, val_losses):
    """Plot losses"""
    n = len(train_losses)
    size = math.ceil(n)
    fig = plt.figure(figsize=(14, size * 5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    if len(val_losses) > 0:
        for i in range(1, n + 1):
            ax = fig.add_subplot(size, 2, i)
            ax.set_title(
                'Loss vs. No. of Epochs (Fold {})'.format(i), fontsize=16)
            ax.plot(train_losses[i - 1], marker="X",
                    color='blue', label='Training')
            ax.plot(val_losses[i - 1], marker="o",
                    color='red', linestyle='--', label='Validation')
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Loss', fontsize=14)
            ax.legend(loc='upper right', fontsize=12)
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Training Loss vs. No. of Epochs', fontsize=16)
        ax.plot(train_losses[0], marker="X", color='blue')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)

    plt.show()
    fig.savefig('loss.pdf')
