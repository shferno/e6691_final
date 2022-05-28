'''
Train Catanet.
Stages:
1) Train CNN (3 epoches) with lcnn
2) Freeze CNN, train RNN (50 epoches) with lrnn
3) Train entire model (10 epoches) with lrnn
4) Freeze CNN, fine-tune RNN (20 epoches) with lrnn
'''
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.logging_utils import timeSince

#! Change the dataset name
from utils.dataset import CataDataset, HerniaDataset
from config import *
from utils.metrics import conf2metrics
from sklearn.metrics import confusion_matrix

import os
import glob
from utils.prepare import windows_path
import platform
import time
import random
import wandb
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ToPILImage, ToTensor, Resize
)


class Trainer(object):
    '''
    Class for training networks.
    '''
    def __init__(self, model, device, pretrain=False, use='cataract', weights=None) -> None:
        '''
        Parameters initialization.
        '''
        self.model = model
        self.device = device
        self.pretrain = pretrain
        self.use = use
        if use == 'cataract':
            self.weights = np.ones(num_labels_cataract)
        else:
            self.weights = weights
        # Define hyper-parameters and optimizer for cnn & rnn
        if pretrain:
            # model is model.cnn
            self.epoch = cnn_epoch
            self.lr = cnn_lr
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            # model is rnn
            self.epoch = rnn_epoch # list
            self.lr = rnn_lr # list
            self.optimizer = []
            for i in range(3):
                self.optimizer.append(
                    optim.Adam(
                        [{'params': self.model.rnn.parameters()},
                        {'params': self.model.cnn.parameters(), 'lr': self.lr[i] / 20}],
                        lr=self.lr[i]
                    )
                )
        # Define loss functions for three different inferences
        self.loss_fc = {
            'phase': self.criterion('phase'),
            'exp': self.criterion('exp'),
            'rsd': self.criterion('rsd')
        }

    def criterion(self, mode:str):
        '''
        Loss functions.
        mode: 'phase', 'exp', 'rsd'
        '''
        num_labels = num_labels_cataract if self.use == 'cataract' else num_labels_hernia
        if mode == 'exp':
            return nn.CrossEntropyLoss()
        elif mode == 'phase':
            label_sum = np.zeros(num_labels)
            #! Change pth for windows
            if self.use == 'cataract':
                if self.pretrain:
                    label_pth = glob.glob(os.path.join(cataract_base_pth, 'train', '**', '*.csv')) # change from data_base_pth to cataract_base_pth
                else:
                    label_pth = sorted(glob.glob(os.path.join(cataract_base_pth, 'train','**','*.csv')))
                fmt_path = lambda x: windows_path(x) if platform.platform().startswith('Win') else x
                for fname_label in fmt_path(label_pth):
                    labels = np.genfromtxt(fname_label, delimiter=',', skip_header=1)[:, 1]
                    for l in range(num_labels):
                        label_sum[l] += np.sum(labels==l)
                loss_weights = 1 / label_sum
                loss_weights[label_sum == 0] = 0.0
                loss_weights = torch.tensor(loss_weights / np.max(loss_weights)).float().to(device)
            else:
                loss_weights = torch.tensor(self.weights).float().to(device)
            return nn.CrossEntropyLoss(weight=loss_weights)
        else:
            return nn.L1Loss()
    
    def train(self, dataLoader:dict, use='cataract') -> dict:
        '''
        Training function for ResNet.
        Input args:
            dataLoader:
            When pretraining CNN: {'train': training data, 'val': validation data}
            When training RNN: {key:{} for key in train_stages}
        Outputs: 
            Dictionary of training history
        '''
        model = self.model
        device = self.device
        pretrain = self.pretrain
        num_epoch = self.epoch
        optimizer = self.optimizer
        loss_fc = self.loss_fc
        num_labels = num_labels_cataract if self.use == 'cataract' else num_labels_hernia

        ############################# Train CNN #############################
        if pretrain:
            print('Start pretraining CNN:')
            
            best_loss_test = np.Infinity
            hist_train_loss = []
            hist_valid_loss = []

            start_time = time.time()
            for epoch in range(num_epoch):
                print('Epoch {}'.format(epoch))
                all_loss_train = torch.zeros(0).to(device)
                model.train()
                # Training data: forward and back-propagationsd
                for ii, (img, labels) in enumerate(dataLoader['train']):
                    print('{}/{}'.format(ii, len(dataLoader['train'])), end='\r')
                    img = img.to(device)  # input data
                    phase_label = labels[:, 0].long().to(device)
                    exp_label = labels[:, 2].long().to(device) - 1
                    with torch.set_grad_enabled(True):
                        phase_pred, exp_pred = model(img)
                        loss = loss_fc['phase'](phase_pred, phase_label) + loss_fc['exp'](exp_pred, exp_label)
                        # update weights
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    all_loss_train = torch.cat((all_loss_train, loss.detach().view(1, -1)))

                    # Interval validation
                    if ii % cnn_val_step == 0:
                        model.eval()
                        with torch.no_grad():
                            val_subepoch_loss = torch.zeros(0).to(device)
                            conf_mat_phase = np.zeros((num_labels, num_labels))
                            conf_mat_exp = np.zeros((2, 2))
                            for _, (img, label) in enumerate(dataLoader['val']):  # for each of the batches
                                img = img.to(device)  # input data
                                phase_label = label[:, 0].long().to(device)
                                exp_label = (label[:, 2] - 1).long().to(device)
                                phase_pred, exp_pred = model(img)  # [batch size, n_classes]
                                loss = loss_fc['phase'](phase_pred, phase_label) + loss_fc['exp'](exp_pred, exp_label)
                                val_subepoch_loss = torch.cat((val_subepoch_loss, loss.detach().view(1, -1)))
                                # argmax and confusion matrix
                                phase_prediction = torch.argmax(phase_pred.detach(), dim=1).cpu().numpy()
                                conf_mat_phase += confusion_matrix(phase_label.cpu().numpy(), phase_prediction,
                                                                    labels=np.arange(num_labels))
                                exp_prediction = torch.argmax(exp_pred.detach(), dim=1).cpu().numpy()
                                conf_mat_exp += confusion_matrix(exp_label.cpu().numpy(), exp_prediction,
                                                                    labels=np.arange(2))
                        # compute metrics
                        val_subepoch_loss = val_subepoch_loss.cpu().numpy().mean()
                        print('val loss: {0:.4f}'.format(val_subepoch_loss), end='')
                        wandb.log({'/val/loss': val_subepoch_loss})
                        hist_valid_loss.append(val_subepoch_loss)
                        # save model at best loss
                        if val_subepoch_loss < best_loss_test:
                            state_dict = model.state_dict()
                            best_loss_test = val_subepoch_loss
                            print('!!')
                            torch.save(
                                {'epoch': epoch + 1,'model_dict': state_dict},
                                os.path.join(cnn_pth, 'cnn.pth')
                            )
                        else:
                            print('')
                        model.train()
                all_loss_train = all_loss_train.cpu().numpy().mean()
                wandb.log({'epoch': epoch, '/train/loss': all_loss_train})
                hist_train_loss.append(all_loss_train)
                print(
                    '%s ([%d/%d] %d%%), train loss: %.4f' %\
                   (timeSince(start_time, (epoch+1) / num_epoch),
                    epoch + 1, num_epoch , (epoch + 1) / num_epoch * 100,
                    all_loss_train)
                )
            print('Finished CNN pretrain')
            
            return {
            'train_loss': hist_train_loss,
            'valid_loss': hist_valid_loss
            }
        
        ############################# Train RNN #############################
        else:
            print('Start training RNN:')
            
            training_stages = train_stages
            remaining_stages = train_stages

            start_time = time.time()
            non_improving_val_counter = 0
            features = {}
            
            hist_train_loss = []
            hist_valid_loss = []
            hist_val_acc = []
            hist_val_recall = []
            hist_val_f1 = []
            hist_exp_acc = []
            
            for step_count, training_step in enumerate(training_stages):
                print(training_step) # 'train_rnn', 'train_all', 'train_rnn'
                if step_count > 0:
                    # load rnn model
                    checkpoint = torch.load(os.path.join(rnn_pth, 'rnn.pth'), map_location=device)
                    model.load_state_dict(checkpoint['model_dict'], strict=False) # add strict
                
                best_loss_on_val = np.Infinity
                stop_epoch = num_epoch[step_count]
                optim = optimizer[step_count]
                if training_step == 'train_rnn':
                    # pre-compute features
                    if len(features)==0:
                        sequences = list(
                            zip(dataLoader['train']['label']+dataLoader['val']['label'],
                            dataLoader['train']['video']+dataLoader['val']['video'])
                        )
                        print('')
                        for ii,(label_path, input_path) in enumerate(sequences):
                            print('{}/{}'.format(ii, len(sequences), end='\r'))
                            if use == 'cataract':
                                data = CataDataset([input_path], [label_path], img_transform)
                                loader = DataLoader(data, batch_size=500, shuffle=False, num_workers=2, pin_memory=True)
                            else:
                                data = HerniaDataset(input_path, label_path[0], label_path[1], transforms=Compose([
                                    RandomResizedCrop(size=(input_size[0], input_size[1]), scale=(0.4, 1.0), ratio=(1.0, 1.0)),
                                    RandomHorizontalFlip(),
                                    RandomVerticalFlip()
                                ]))
                                loader = DataLoader(data, batch_size=train_all_batch, shuffle=False, num_workers=2, pin_memory=True)
                            features[input_path] = []
                            for _, (X, _) in enumerate(loader):
                                with torch.no_grad():
                                    features[input_path].append(model.cnn(X.float().to(device)).cpu().numpy())
                            features[input_path] = np.concatenate(features[input_path])
                    model.freeze_cnn(True)
                    model.freeze_rnn(False)
                elif training_step == 'train_all':
                    model.freeze_cnn(False)
                    model.freeze_rnn(False)
                    features = {}
                else:
                    raise RuntimeError('training step {0} not implemented'.format(training_step))

                for epoch in range(stop_epoch):
                    all_precision = {}
                    average_precision = {}
                    all_recall = {}
                    average_recall = {}
                    all_f1 = {}
                    average_f1 = {}
                    accuracy_exp = {}
                    conf_mat_phase = {key: np.zeros((num_labels, num_labels)) for key in ['val']}
                    conf_mat_exp = {key: np.zeros((2, 2)) for key in ['val']}
                    all_loss = {key: torch.zeros(0).to(device) for key in ['train', 'val']}
                    all_loss_phase = {key: torch.zeros(0).to(device) for key in ['train', 'val']}
                    all_loss_experience = {key: torch.zeros(0).to(device) for key in ['train', 'val']}
                    all_loss_rsd = {key: torch.zeros(0).to(device) for key in ['train', 'val']}

                    for phase in ['train', 'val']: #iterate through both training and validation states
                        sequences = list(zip(dataLoader[phase]['label'], dataLoader[phase]['video']))
                        if phase == 'train':
                            model.train()  # Set model to training mode
                            random.shuffle(sequences)  # random shuffle training sequences
                            model.cnn.eval()  # required due to batch-norm, even when training end-to-end
                        else:
                            model.eval()   # Set model to evaluate mode

                        for ii, (label_path, input_path) in enumerate(sequences):
                            if (training_step == 'train_rnn'):
                                label = torch.tensor(np.genfromtxt(label_path, delimiter=',', skip_header=1)[:, 1:])
                                loader = [(torch.tensor(features[input_path]).unsqueeze(0),
                                            label[:len(features[input_path]),:])]
                                skip_features = True
                            else:
                                batch_size = train_all_batch
                                if use == 'cataract':
                                    data = CataDataset([input_path], [label_path], img_transform=img_transform)
                                    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
                                else:
                                    data = HerniaDataset(input_path, label_path[0], label_path[1], transforms=Compose([
                                        RandomResizedCrop(size=(input_size[0], input_size[1]), scale=(0.4, 1.0), ratio=(1.0, 1.0)),
                                        RandomHorizontalFlip(),
                                        RandomVerticalFlip()
                                    ]))
                                    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
                                skip_features = False

                            for i, (X, y) in enumerate(loader):
                                if len(y.shape) > 2: # batch-size is automatically removed from tensor dimensions for label
                                    y = y.squeeze()
                                y_experience = torch.add(y[:, 2], -1).long().to(device)
                                # Specific to dataset
                                y_rsd = (y[:, 5]/60.0/25.0).float().to(device)
                                y = y[:, 0].long().to(device)
                                X = X.float().to(device)

                                with torch.set_grad_enabled(phase == 'train'):
                                    stateful = (i > 0)
                                    phase_prediction, experience_prediction, rsd_prediction = model(X, stateful=stateful,
                                                                                            skip_features=skip_features)
                                    loss_step = loss_fc['phase'](phase_prediction, y)
                                    loss_experience = loss_fc['exp'](experience_prediction, y_experience)
                                    rsd_prediction = rsd_prediction.squeeze(1)
                                    loss_rsd = loss_fc['rsd'](rsd_prediction, y_rsd)
                                    loss = loss_step + 0.3 * loss_experience + loss_rsd
                                    if phase == 'train':  # in case we're in train mode, need to do back propagation
                                        optim.zero_grad()
                                        loss.backward()
                                        optim.step()

                                    all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))
                                    all_loss_phase[phase] = torch.cat((all_loss_phase[phase], loss_step.detach().view(1, -1)))
                                    all_loss_experience[phase] = torch.cat((all_loss_experience[phase], loss_experience.detach().view(1, -1)))
                                    all_loss_rsd[phase] = torch.cat((all_loss_rsd[phase], loss_rsd.detach().view(1, -1)))
                                if phase in ['val']:
                                    hard_prediction = torch.argmax(phase_prediction.detach(), dim=1).cpu().numpy()
                                    hard_prediction_exp = torch.argmax(experience_prediction.detach(), dim=1).cpu().numpy()
                                    conf_mat_phase[phase] += confusion_matrix(y.cpu().numpy(), hard_prediction, labels=np.arange(num_labels))
                                    conf_mat_exp[phase] += confusion_matrix(y_experience.cpu().numpy(), hard_prediction_exp, labels=np.arange(2))

                        all_loss[phase] = all_loss[phase].cpu().numpy().mean()
                        all_loss_phase[phase] = all_loss_phase[phase].cpu().numpy().mean()
                        all_loss_experience[phase] = all_loss_experience[phase].cpu().numpy().mean()
                        all_loss_rsd[phase] = all_loss_rsd[phase].cpu().numpy().mean()
                        if phase in ['val']:
                            precision, recall, f1, accuracy = conf2metrics(conf_mat_phase[phase])
                            accuracy_exp[phase] = conf2metrics(conf_mat_exp[phase])[3]
                            all_precision[phase] = precision
                            all_recall[phase] = recall
                            average_precision[phase] = np.mean(all_precision[phase])
                            average_recall[phase] = np.mean(all_recall[phase])
                            all_f1[phase] = f1
                            average_f1[phase] = np.mean(all_f1[phase])
                        # Add log
                        log_epoch = step_count*epoch+epoch
                        wandb.log({'epoch': log_epoch, f'{phase}/loss': all_loss[phase],
                                f'{phase}/loss_rsd': all_loss_rsd[phase],
                                f'{phase}/loss_step': all_loss_phase[phase],
                                f'{phase}/loss_exp': all_loss_experience[phase]})
                        if ((epoch % rnn_interval) == 0) & (phase in ['val']):
                            wandb.log({'epoch': log_epoch, f'{phase}/precision': average_precision[phase],
                                    f'{phase}/recall': average_recall[phase], f'{phase}/f1': average_f1[phase],
                                    f'{phase}/exp_acc': accuracy_exp[phase]})
                    
                    log_text = '%s ([%d/%d] %d%%), train loss: %.4f val loss: %.4f lp: %.4f le: %.4f' % \
                            (timeSince(start_time, (epoch + 1) / stop_epoch),
                                epoch + 1, stop_epoch, (epoch + 1) / stop_epoch * 100,
                                all_loss['train'], all_loss['val'], all_loss_phase['val'], all_loss_experience['val'])
                    log_text += ' val precision: {0:.4f}, recall: {1:.4f}, f1: {2:.4f}, acc_exp: {3:.4f}'.format(average_precision['val'],
                                                                                            average_recall['val'],
                                                                                            average_f1['val'],
                                                                                                accuracy_exp['val'])
                    print(log_text, end='')
                    # Store history
                    hist_train_loss.append(all_loss['train'])
                    hist_valid_loss.append(all_loss['val'])
                    hist_val_acc.append(average_precision['val'])
                    hist_val_recall.append(average_recall['val'])
                    hist_val_f1.append(average_f1['val'])
                    hist_exp_acc.append(accuracy_exp['val'])
                    
                    if all_loss["val"] < best_loss_on_val:
                        # if current loss is the best we've seen, save model state
                        non_improving_val_counter = 0
                        best_loss_on_val = all_loss["val"]
                        print('  **')
                        state = {'epoch': epoch + 1,
                                'model_dict': model.state_dict(),
                                'remaining_steps': remaining_stages}

                        torch.save(state, os.path.join(rnn_pth, 'rnn.pth'))
                        # Add log
                        wandb.summary['best_epoch'] = epoch + 1
                        wandb.summary['best_loss_on_val'] = best_loss_on_val
                        wandb.summary['f1'] = average_f1['val']
                        wandb.summary['exp_acc'] = accuracy_exp['val']
                    else:
                        print('')
                        non_improving_val_counter += 1
                remaining_stages.pop(0)
            print('Finished training RNN.')

            return {
                'train_loss': hist_train_loss,
                'valid_loss': hist_valid_loss,
                'val_precision': hist_val_acc,
                'val_recall': hist_val_recall,
                'val_f1': hist_val_f1,
                'exp_acc': hist_exp_acc
            }

