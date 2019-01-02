import sys
import numpy as np
import torch
from torchvision import datasets, transforms

class cifar10_trainer():
    def __init__(self, model, device, batch_size=64):
        self.model = model
        self.device = device

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.train_loader = torch.utils.data.DataLoader(
                                datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor(),
                                    self.normalize,
                                ])),
                                batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
                                datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    self.normalize,
                                ])),
                                batch_size=batch_size, shuffle=True)

        self.loss_module = torch.nn.CrossEntropyLoss()

    def accuracy(self, predictions, targets):
        targets = targets.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        return np.sum(1. * (np.argmax(predictions, axis=1) == targets)) / targets.shape[0]

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            y_pred = self.model.forward(data)
            loss = self.loss_module(y_pred, target)

            #Take optimizer step
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            train_loss += loss.item()
        return train_loss / (batch_idx + 1)
    
    def test(self):
        self.model.eval()
        test_loss, test_acc = 0, 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            y_pred = self.model.forward(data)
            test_loss += self.loss_module(y_pred, target).item() 
            test_acc += self.accuracy(y_pred, target)
        return test_loss / (batch_idx + 1), test_acc / (batch_idx + 1)

    def train_model(self, epochs, eval_freq=1):
        print("Training the", self.model.name,"for", epochs, "epochs!")
        epoch_list, train_losses, test_losses, test_accs = [], [], [], []

        for e in range(epochs):
            train_loss = self.train()
            if (e + 1) % eval_freq == 0:
                test_loss, test_acc = self.test()
                sys.stdout.write('\rTraining the model for epoch: %d / %d | train loss: %f | test loss: %f | test_acc: %f' % (e + 1, epochs, train_loss, test_loss, test_acc))
                sys.stdout.flush()

                #Append statistics
                epoch_list.append(e)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
        print("\n")
        return epoch_list, train_losses, test_losses, test_accs
