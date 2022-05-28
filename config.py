'''
Configurations.
'''
import torch
import os
from torchvision import transforms

# Use GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data Properties
# Number of labels
num_labels_cataract = 11 # Cataract_101
num_labels_hernia = 14

# Path
# Path to data
cataract_base_pth = 'data/cataract101'
hernia_base_pth = 'data/videos'
hernia_image_pth ='D:/e6691/6691_assignment2/images'
hernia_label_pth = 'data/labels/video.phase.trainingData.clean.StudentVersion.csv'
hernia_name_pth = 'data/labels/all_labels_hernia.csv'
alpha_smooth_class = 0.1
# baseline
baseline = 'resnet18'
# Output of CNN
#cnn_pth = 'output/cnn_svrc.pth'
# cataract
#cnn_pth = 'E:/e6691-2022spring-project-ccsz/output'
# hernia
cnn_pth = 'E:/e6691-2022spring-project-ccsz/output/hernia'
# Output of RNN
#rnn_pth = 'output/rnn_svrc.pth'
# cataract
#rnn_pth = 'E:/e6691-2022spring-project-ccsz/output'
# hernia
rnn_pth = 'E:/e6691-2022spring-project-ccsz/output/hernia'
# Path to log
log_pth = 'logs'

# Hyper parameters
# Input Size (list)
input_size = [224,224]
# Train CNN
cnn_lr = 1e-3
cnn_epoch = 1 # change from 3 to 1
cnn_val_step = 100
cnn_train_batch = 50
cnn_val_batch = 150
# Train RNN
rnn_epoch = [1, 1, 1]
rnn_lr = [0.001, 0.005, 0.001]
train_stages = ['train_rnn', 'train_all', 'train_rnn']
train_all_batch = 48
rnn_interval = 1

# Train SVRC
CNN_BATCH = 32
LSTM_BATCH = 32
SAMPLING_RATE = 0.1

# Transform
img_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((input_size[0], input_size[1])),
                                    transforms.ToTensor()])

data_transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(size=(input_size[0], input_size[1]), scale=(0.4, 1.0), ratio=(1.0, 1.0))
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size[0], input_size[1]))
    ])
}