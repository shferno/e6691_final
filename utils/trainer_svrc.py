'''
Train Catanet.
Stages:
1) Train CNN (3 epoches) with lcnn
2) Freeze CNN, train RNN (50 epoches) with lrnn
3) Train entire model (10 epoches) with lrnn
4) Freeze CNN, fine-tune RNN (20 epoches) with lrnn
'''
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, WeightedRandomSampler

#! Change the dataset name
from utils.dataset import SVRCDataset
from config import *
from utils.metrics import conf2metrics
from sklearn.metrics import confusion_matrix

from utils.prepare import windows_path


class ResnetTrainVal(object):
    '''
    Class for training Resnet.
    '''
    def __init__(self, model, device, EPOCH, BATCH_SIZE, LR, class_weights, weights) -> None:
        '''
        ResNet training parameters initialization.
        '''
        self.model = model
        self.device = device
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        self.class_weights = class_weights
        self.weights = weights
        self.phase_loss = self.get_criterion('phase')
        self.rsd_loss = self.get_criterion('rsd')


    def get_criterion(self, mode:str):
        '''
        Loss functions.
        mode: 'phase', 'exp', 'rsd'
        '''
        if mode == 'exp':
            return nn.CrossEntropyLoss()
        elif mode == 'phase':
            loss_weights = 1 / torch.tensor(self.class_weights).float().to(device)
            return nn.CrossEntropyLoss(weight=loss_weights)
        else:
            return nn.L1Loss()


    def train(self, labels, features, validation:tuple, transform, path) -> dict: #, val_ratio=0.7):
        '''
        Training function for ResNet.
        Input args:
            labels, features: Traning data path
            validation: Validation data path
            transform: Data transformation chosen
            path: Path for saving optimal model weights
        Outputs: 
            Dictionary of training history
        '''
        print('Training {} CNN: '.format(baseline))

        # TRAIN_SIZE = int(val_ratio * len(features))
        # TEST_SIZE = len(features) - TRAIN_SIZE

        train = SVRCDataset(features, labels, transform['train'])
        test = SVRCDataset(validation[0], validation[1], transform['valid'])
        #train, test = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])
        print('length of train:', len(train))
        print('length of validation', len(test))

        train_loader = DataLoader(train, self.BATCH_SIZE, sampler=WeightedRandomSampler(
            self.weights, len(train)
        ))
        test_loader = DataLoader(test, self.BATCH_SIZE, shuffle=True)

        self.model.pretrain = True

        hist_train_loss = []
        hist_train_acc = []
        hist_train_mae = []
        hist_valid_loss = []
        hist_valid_acc = []
        hist_valid_mae = []

        for epoch in range(self.EPOCH):
            self.model.train()

            train_loss = 0.0
            train_acc = 0.0
            train_mae = 0.0

            for i, data in enumerate(train_loader):
                print('\r{}/{}'.format(i + 1, len(train_loader)), end='')

                features = data['feature'].float()
                labels = data['label']
                phase = labels[:, 0].type(torch.LongTensor)
                rsd = labels[:, 1]
                features, phase, rsd = features.to(self.device), phase.to(self.device), rsd.to(self.device)

                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()
                    phase_preds, rsd_preds = self.model(features)
                    loss_rsd = self.rsd_loss(rsd_preds.squeeze(), rsd)
                    loss = self.phase_loss(phase_preds.squeeze(), phase) #+ loss_rsd
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()
                phase_preds = torch.max(phase_preds.data, 1)[1]
                train_acc += (phase_preds==phase).sum().item()
                train_mae += loss_rsd

            torch.save(self.model.state_dict(),os.path.join(path,'cnn.pth'))
            train_loss /= len(train)
            train_acc /= len(train)
            train_mae /= len(train)
            hist_train_loss.append(train_loss)
            hist_train_acc.append(train_acc)
            hist_train_mae.append(train_mae)


            valid_loss = 0.0
            valid_acc = 0.0
            valid_mae = 0.0
            total = 0

            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    features = data['feature']
                    labels = data['label']
                    phase = labels[:, 0].type(torch.LongTensor)
                    rsd = labels[:, 1]

                    features, phase, rsd = features.to(self.device), phase.to(self.device), rsd.to(self.device)
                    phase_preds, rsd_preds = self.model(features)
                    loss_rsd = self.rsd_loss(rsd_preds.squeeze(), rsd)
                    loss = self.phase_loss(phase_preds.squeeze(), phase) #+ loss_rsd
                    valid_loss += loss.item()

                    phase_preds = torch.max(phase_preds.data, 1)[1]
                    valid_acc += (phase_preds==phase).sum().item()
                    valid_mae += loss_rsd
                    total += features.size(0)

            valid_loss /= len(test)
            valid_acc /= len(test)
            valid_mae /= len(test)
            hist_valid_loss.append(valid_loss)
            hist_valid_acc.append(valid_acc)
            hist_valid_mae.append(valid_mae)

            print(
                f'\rEpoch {epoch+1} Train Loss: {train_loss} Train Acc: {train_acc} Train MAE: {train_mae}'
                f' || Valid Loss: {valid_loss} Valid Acc: {valid_acc} Valid MAE: {valid_mae}'
            )

        return {
            'train_loss': hist_train_loss,
            'train_acc': hist_train_acc,
            'train_mae': hist_train_mae,
            'valid_loss': hist_valid_loss,
            'valid_acc': hist_valid_acc,
            'valid_mae': valid_mae
        }


class LstmTrainVal(object):
    '''
    Class for training SVRCNet.
    '''
    def __init__(self, model,device, EPOCH, BATCH_SIZE, LR, class_weights, weights) -> None:
        '''
        LSTM training parameters initialization.
        '''
        self.model = model
        self.device = device
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        self.class_weights = class_weights
        self.weights = weights
        self.phase_loss = self.get_criterion('phase')
        self.rsd_loss = self.get_criterion('rsd')


    def get_criterion(self, mode: str):
        '''
        Loss functions.
        mode: 'phase', 'exp', 'rsd'
        '''
        if mode == 'exp':
            return nn.CrossEntropyLoss()
        elif mode == 'phase':
            loss_weights = 1 / torch.tensor(self.class_weights).float().to(device)
            return nn.CrossEntropyLoss(weight=loss_weights)
        else:
            return nn.L1Loss()


    def train(self, labels, features, validation:tuple, transform, path, eval_intval=3):
        '''
        Training function for SVRCNet.
        Inputs:
            Similar to the trainer for LSTM.
            eval_intval: Step for validation (One validation for each eval_intval epochs)
        Outputs:
            Dictionary of training history
        '''
        print('Training {} RNN: '.format(baseline))

        dataset = SVRCDataset(features, labels, transform['train'])
        data_loader = DataLoader(
            dataset, batch_sampler=BatchSampler(
                SequentialSampler(dataset),
                self.BATCH_SIZE,
                drop_last=True
            )
        )
        valid_set = SVRCDataset(validation[0], validation[1], transform['valid'])
        valid_loader = DataLoader(
            valid_set, batch_sampler=BatchSampler(
                SequentialSampler(valid_set), 
                self.BATCH_SIZE, 
                drop_last=True
            )
        )
        print('length of train:', len(dataset))
        print('length of validation', len(valid_set))

        self.model.pretrain = False

        hist_train_loss = []
        hist_train_acc = []
        hist_train_mae = []
        hist_valid_loss = []
        hist_valid_acc = []
        hist_valid_mae = []

        for epoch in range(self.EPOCH):
            self.model.rnn.train()
            self.model.fc1.train()
            self.model.fc2.train()

            train_loss = 0.0
            train_acc = 0.0
            train_mae = 0.0

            for i, data in enumerate(data_loader):
                print('\r{}/{}'.format(i + 1, len(data_loader)), end='')
                features = data['feature'].float()
                labels = data['label']
                phase = labels[:, 0].type(torch.LongTensor)
                rsd = labels[:, 1]
                features, phase, rsd = features.to(self.device), phase.to(self.device), rsd.to(self.device)

                with torch.enable_grad():
                    phase_preds, rsd_preds = self.model(features)
                    loss_rsd = self.rsd_loss(rsd_preds.squeeze(), rsd)
                    loss = loss = self.phase_loss(phase_preds.squeeze(), phase) + 0.1 * loss_rsd

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()
                phase_preds = torch.max(phase_preds.data, 1)[1]
                train_acc += (phase_preds==phase).sum().item()
                train_mae += loss_rsd

            torch.save(self.model.state_dict(),os.path.join(path,'rnn_{}.pth'.format(time.time())))


            train_loss /= len(dataset)
            train_acc /= len(dataset)
            train_mae /= len(dataset)
            hist_train_loss.append(train_loss)
            hist_train_acc.append(train_acc)
            hist_train_mae.append(train_mae)

            valid_loss = 0.0
            valid_acc = 0.0
            valid_mae = 0.0

            if (epoch + 1) % eval_intval == 0:
                total = 0
                self.model.eval()
                with torch.no_grad():
                    for i, data in enumerate(valid_loader):
                        features = data['feature']
                        labels = data['label']
                        phase = labels[:, 0].type(torch.LongTensor)
                        rsd = labels[:, 1]

                        features, phase, rsd = features.to(self.device), phase.to(self.device), rsd.to(self.device)
                        phase_preds, rsd_preds = self.model(features)
                        loss_rsd = self.rsd_loss(rsd_preds.squeeze(), rsd)
                        loss = self.phase_loss(phase_preds.squeeze(), phase) + 0.1 * loss_rsd
                        valid_loss += loss.item()

                        phase_preds = torch.max(phase_preds.data, 1)[1]
                        valid_acc += (phase_preds==phase).sum().item()
                        valid_mae += loss_rsd
                        total += features.size(0)

                valid_loss /= len(valid_set)
                valid_acc /= len(valid_set)
                valid_mae /= len(valid_set)
                hist_valid_loss.append(valid_loss)
                hist_valid_acc.append(valid_acc)
                hist_valid_mae.append(valid_mae)

            print(
                f'\rEpoch {epoch + 1} Train Loss: {train_loss} Train Acc: {train_acc} Train MAE: {train_mae}'
                f' || Valid Loss: {valid_loss} Valid Acc: {valid_acc} Valid MAE: {valid_mae}'
            )

        return {
            'train_loss': hist_train_loss,
            'train_acc': hist_train_acc,
            'train_mae': hist_train_mae,
            'valid_loss': hist_valid_loss,
            'valid_acc': hist_valid_acc,
            'valid_mae': valid_mae
        }


class Evaluator:
    '''
    Evaluate dataset without labels.
    '''
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, images, transform, pretrain):
        '''
        Predict input data.
        Input args:
            images: Path for evaluating data.
            transform: Chosen data transformation method
            pretrain: ResNet or SVRCNet mode
        '''
        dataset = SVRCDataset(images, None, transform)
        loader = DataLoader(dataset, batch_size=3, drop_last=True)
        preds = []
        self.model.pretrain = pretrain
        self.model.eval()
        for i,data in enumerate(loader):
            feature = data['feature'].float().to(self.device)
            pred = torch.max(self.model(feature).data, 1)[1]
            preds.append(pred)
        return sum(list(map(torch.Tensor.tolist, preds)), [])

    def eval(self, preds, labels):
        ''' 
        Evaluate the predictions with ground truth labels. 
        There should not be batched in preds and labels
        '''
        acc = sum([p == l for p,l in zip(preds, labels)]) / len(labels)
        print('Accuracy: {}'.format(acc))
        return acc

