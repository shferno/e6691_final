from sqlite3 import DatabaseError
from utils.trainer import Trainer
from utils.dataset import CataDataset, HerniaDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.CNN_model import CNN_model
from models.RNN_model import RNN_model
from models.svrc import SVRC
from utils.trainer_svrc import ResnetTrainVal, LstmTrainVal
from config import *
from utils.prepare import prepare_data, get_class_weights, get_label_weights, windows_path, read

import os
import glob
import time
import platform
import wandb
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ToPILImage, ToTensor, Resize
)


def train_cnn(use='cataract'):
    '''
    Train CNN.
    ---
    Parameter
    - use: str = ['cataract'|'hernia'].
        Select dataset to train on. 
    '''
    # make output dir for cnn model
    os.makedirs(cnn_pth, mode=0o777, exist_ok=True)
    os.chmod(cnn_pth, 0o777)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # logging
    run = wandb.init(project='{}_rsd'.format(use), group='catnet')
    run.config.data = cataract_base_pth
    run.name = run.id
   
    # data preprocessing
    img_transform = {}
    img_transform['train'] = Compose([
        ToPILImage(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomResizedCrop(size=(input_size[0], input_size[1]), scale=(0.4, 1.0), ratio=(1.0, 1.0)),
        ToTensor()
    ])
    img_transform['val'] = Compose([ToPILImage(), Resize((input_size[0], input_size[1])), ToTensor()])

    # load data
    dataLoader = {}
    config = {'train': {}, 'val': {}}
    config['train']['batch_size'] = cnn_train_batch
    config['val']['batch_size'] = cnn_val_batch
    for phase in ['train', 'val']:
        data_folders = sorted(glob.glob(os.path.join(cataract_base_pth, phase, '*')))
        labels = sorted(glob.glob(os.path.join(cataract_base_pth, phase, '**', '*.csv')))
        fmt_path = lambda x: windows_path(x) if platform.platform().startswith('Win') else x
        if use == 'cataract':
            dataset = CataDataset(fmt_path(data_folders), img_transform=img_transform[phase], label_files=fmt_path(labels))
            dataLoader[phase] = DataLoader(dataset, batch_size=config[phase]['batch_size'],shuffle=(phase=='train'), \
                num_workers=2, pin_memory=True)
        else:
            videos, labels_df, all_labels_name = prepare_data(hernia_base_pth, hernia_label_pth, hernia_name_pth)
            img_trans = Compose([
                RandomResizedCrop(size=(input_size[0], input_size[1]), scale=(0.4, 1.0), ratio=(1.0, 1.0)),
                RandomHorizontalFlip(),
                RandomVerticalFlip()
            ])
            dataset = HerniaDataset(videos, labels_df, all_labels_name, img_trans)
            class_weights = get_class_weights(labels_df, alpha_smooth_class)
            weights = get_label_weights(labels_df, all_labels_name, class_weights, len(dataset))
            dataLoader[phase] = DataLoader(dataset, batch_size=config[phase]['batch_size'], \
                sampler=WeightedRandomSampler(weights, len(dataset), replacement=True), num_workers=2, pin_memory=True)

    # load model
    cnn_model = CNN_model('densenet169', n_step_classes=num_labels_cataract if use == 'cataract' else num_labels_hernia)
    cnn_model = cnn_model.to(device)

    # train
    cnn_trainer = Trainer(cnn_model, device, pretrain=True, use=use, weights=class_weights.values)
    hist = cnn_trainer.train(dataLoader)
    return hist

def train_rnn(use='cataract'):
    # train rnn
    # make output dir for rnn model
    os.makedirs(rnn_pth, mode=0o777, exist_ok=True)
    os.chmod(rnn_pth, 0o777)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loggging
    run = wandb.init(project='cataract_rsd', group='catnet')
    run.name = run.id

    # data processing
    print('collect rnn training dataset:')
    # glob all video files and perform forward pass through CNN. Store the features in npy-arrays for RNN training
    # --- glob data set
    sequences_path = {key:{} for key in ['train', 'val']}
    # foramt path for different systems
    fmt_path = lambda x: windows_path(x) if platform.platform().startswith('Win') else x
    if use == 'cataract':
        sequences_path['train']['label'] = fmt_path(sorted(glob.glob(os.path.join(cataract_base_pth, 'train','**','*.csv'))))
        sequences_path['val']['label'] = fmt_path(sorted(glob.glob(os.path.join(cataract_base_pth, 'val', '**', '*.csv'))))
        sequences_path['train']['video'] = fmt_path(sorted(glob.glob(os.path.join(cataract_base_pth, 'train', '*/'))))
        sequences_path['val']['video'] = fmt_path(sorted(glob.glob(os.path.join(cataract_base_pth, 'val', '*/'))))
    else:
        videos, labels_df, all_labels_name = prepare_data(hernia_base_pth, hernia_label_pth, hernia_name_pth)
        sequences_path['train']['label'] = [(labels_df, all_labels_name)] * len(videos)
        sequences_path['val']['label'] = []
        sequences_path['train']['video'] = videos
        sequences_path['val']['video'] = []

    print('number of sequences: train {0}, val {1}'.format(len(sequences_path['train']['label']),
                                                           len(sequences_path['val']['label'])))

    # load model
    model = RNN_model('densenet169', n_step_classes=num_labels_cataract if use == 'cataract' else num_labels_hernia)
    model.cnn.load_state_dict(torch.load(os.path.join(cnn_pth, 'cnn.pth')), strict = False)
    #model.set_cnn_as_feature_extractor()
    model = model.to(device)

    # train
    rnn_trainer = Trainer(model,device,pretrain=False)
    hist = rnn_trainer.train(sequences_path)

    return hist


def train(y:list, X:list, validation:tuple, class_weights, weights, pretrain=True) -> dict:
    '''
    Train SVRCNet model and saved the best model.
    Input args:
        y: Labels from training data
        X: Features from training data
        validation: Choose validataion data set
        pretrain: ResNet pretrain mode when pretrain=True; SVRC model when pretrain=False
    Output:
        hist: Dictionary of training and validation history
    '''
    model = SVRC(baseline)
    model.pretrain = pretrain
    if torch.cuda.is_available():
        model.to(device)

    start_time = time.time()
    if pretrain == True:
        trainer = ResnetTrainVal(model, device,  EPOCH=3, BATCH_SIZE=CNN_BATCH, LR=1e-3, class_weights=class_weights, weights=weights)
        hist = trainer.train(y, X, validation, data_transform, path=cnn_pth) #, val_ratio=0.7)
        with open(os.path.join(log_pth, 'cnn_{}.txt'.format(time.time())), 'w') as f:
            f.write(str(hist))
    else:
        model.load_state_dict(torch.load(os.path.join(cnn_pth,'cnn.pth'), map_location=device), strict=False)
        trainer = LstmTrainVal(model, device, EPOCH=5, BATCH_SIZE=LSTM_BATCH, LR=1e-5, class_weights=class_weights, weights=weights)
        hist = trainer.train(y,X, validation, transform=data_transform, path=rnn_pth, eval_intval=1)
        with open(os.path.join(log_pth, 'rnn_{}.txt'.format(time.time())), 'w') as f:
            f.write(str(hist))

    #path += str(int(time.time()))

    end_time = time.time()
    print('Time:{:.2}min'.format((end_time-start_time)/60.0))

    return hist

def test(y, X, weights, batch, pretrain = False) -> list:
    '''
    Test SVRCNet model.
    Input args:
        y: Ground truth labels (y = None when labels are unknown) from test data
        X: Features from test data
        weights: Saved model path
        batch: Test batch size
        pretrain: Same as train function
    Output:
        predicts: Inferred labels from test data
    '''
    predicts = []
    model = SVRC(baseline)
    model.pretrain = pretrain
    if torch.cuda.is_available():
        model.to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    predicts = model.predict(X, y, BATCH_SIZE=batch, transform=data_transform, device=device)
    return predicts

def main():
    '''
    Main exceution function.
        X,y: training data feature, labels
        X_test,y_test: test data feature, labels
        hist_res: history of pretraining ResNet
        preds_res: test results using ResNet
        hist_svrc: history of training SVRCNet
        preds_svrc: test results using SVRCNet
    '''
    num_train = 16
    num_val = 4
    image_paths, phase, elapsed, rsd, class_weights, weights, cnts = read(hernia_base_pth, hernia_label_pth, hernia_name_pth, num_train + num_val)
    num_train_imgs = sum(cnts[:num_train])
    num_val_imgs = sum(cnts[num_train:num_train+num_val])
    X = image_paths[:num_train_imgs]
    y = (phase[:num_train_imgs], elapsed[:num_train_imgs], rsd[:num_train_imgs])
    X_test = image_paths[num_train_imgs:num_train_imgs+num_val_imgs]
    y_test = (phase[num_train_imgs:num_train_imgs+num_val_imgs], elapsed[num_train_imgs:num_train_imgs+num_val_imgs], rsd[num_train_imgs:num_train_imgs+num_val_imgs])
    weights = weights[:num_train_imgs]

    #hist_res = train(y,X,(X_test,y_test),class_weights,weights,pretrain=True)
    #preds_res = test(y_test, X_test, cnn_pth, batch=64, pretrain = True)

    hist_svrc = train(y,X,(X_test,y_test),class_weights,weights,pretrain=False)
    #preds_svrc = test(y_test, X_test, rnn_pth, batch=3)


if __name__ == '__main__':
    #hist_cnn = train_cnn('hernia')
    #hist_rnn = train_rnn()
    torch.cuda.empty_cache()
    main()

