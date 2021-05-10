# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Ishrat Badami (badami.ishrat@gmail.com)
# ------------------------------------------------------------------------------

import copy
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config
from data_utils import RandomCrop, SegmentationDataset
from model import DualResNet_imagenet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pre_trained_model_weights(path):
    cfg = config
    cfg.defrost()
    cfg.MODEL.PRETRAINED = path
    net = DualResNet_imagenet(cfg, pretrained=True)
    return net


def transfer_learning(model):
    model.final_layer.conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
    model.seghead_extra.conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
    return model


def train_model(model, criterion, optimizer, data_loader, n_epochs=50000):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0

        # Iterate over data.
        for data in data_loader:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            # zero the parameter gradients
            for param in model.parameters():
                param.grad = None

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                preds = outputs[0]
                loss = criterion(preds, labels)

                # backward + optimize
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(training_dataset)

            print('training loss: {:.4f}'.format(epoch_loss))

            # deep copy the model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    return best_model_wts


if __name__ == "__main__":

    training_data = './dataset/train.lst'
    path_to_model = "./pretrained_models/best_val_smaller.pth"
    model_save_path = './result/final_state.pth'
    num_epochs = 50000

    training_dataset = SegmentationDataset(dataset_list_file='./dataset/train.lst',
                                           transforms_op=transforms.Compose([
                                               RandomCrop((512)),
                                           ]))

    dataloader = DataLoader(training_dataset, shuffle=True, batch_size=2, drop_last=True)

    net = load_pre_trained_model_weights(path_to_model)
    net = transfer_learning(net)
    net = net.to(device=device)

    loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(0.3))
    adam_optimizer = torch.optim.Adam(params=net.parameters())
    trained_model_weights = train_model(net, loss, adam_optimizer, dataloader, n_epochs=num_epochs)
    torch.save(trained_model_weights, os.path.join(model_save_path))

