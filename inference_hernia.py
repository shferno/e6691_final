'''
File adopted from CataNet https://github.com/aimi-lab/catanet.git
Marafioti A. et al. (2021) CataNet: Predicting Remaining Cataract Surgery Duration. \
In: de Bruijne M. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. \
MICCAI 2021. Lecture Notes in Computer Science, vol 12904. Springer, Cham. 
'''

import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from utils.prepare import read
from utils.trainer import *
import torch
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize
from torch.utils.data import BatchSampler, SequentialSampler
import sys
sys.path.append('../')
from models.svrc import SVRC
import glob
from config import *
from utils.dataset import SVRCDataset
# from sklearn.metrics import confusion_matrix


def main(out, checkpoint):
    '''
    save predcitions in csv
    ------------------------
    out: path to save csv files
    checkpoing: 'path to model checkpoint .pth file.'
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = SVRC('resnet18').to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict = False)
        model_type = 'SVRC'
    except RuntimeError:
        pass
    print(model_type, ' loaded')
    model.eval()
    model = model.to(device)
    image_paths, phase, elapsed, rsd, class_weights, weights, cnts = read(hernia_base_pth, hernia_label_pth, hernia_name_pth, 20, 16)
    num_imgs = sum(cnts[0:4])
    X = image_paths[:num_imgs]
    y = (phase[:num_imgs], elapsed[:num_imgs], rsd[:num_imgs])
    X = image_paths[:]
    y = (phase[:], elapsed[:], rsd[:])
    dataset = SVRCDataset(X, y, data_transform['valid'])
    data_loader = DataLoader(
        dataset, batch_sampler=BatchSampler(
            SequentialSampler(dataset),
            batch_size = 200,
            drop_last=True
        )
    )
    outputs = []
    for i, data in enumerate(data_loader):
        features = data['feature'].float()
        elapsed_time = data['feature'][:,3,0,0].to(device)
        labels = data['label']
        phase = labels[:, 0].type(torch.LongTensor)
        rsd = labels[:, 1]
        features, phase, rsd = features.to(device), phase.to(device), rsd.to(device)
        with torch.no_grad():
            phase_preds, rsd_preds = model(features)
            phase_pred_hard = torch.argmax(phase_preds, dim=-1).view(-1).cpu().numpy()
            rsd_preds = rsd_preds.view(-1).cpu().numpy()
            elapsed_time = elapsed_time.view(-1).cpu().numpy()
            rsd_gt = rsd.clone().cpu().numpy()
            phase_gt = phase.clone().cpu().numpy()
            outputs.append(np.asarray([elapsed_time, rsd_preds, phase_pred_hard, rsd_gt, phase_gt]).T)
    outputs = np.concatenate(outputs,axis = 0)
    np.savetxt(os.path.join(out, 'output_resnet224.csv'),
                   outputs, delimiter=',',
                   header='elapsed,predicted_rsd,predicted_step,rsd,phase', comments='')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--out',
        type=str,
        default='E:/e6691-2022spring-project-ccsz/hernia_data',
        help='path to output folder.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default = 'E:/e6691-2022spring-project-ccsz/hernia_data',
        help='path to processed video file or multiple files.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default = 'E:/e6691-2022spring-project-ccsz/output/resnet224/rnn_1652205228.8080022.pth',
        help='path to model checkpoint .pth file.'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    main(out=args.out, input=args.input, checkpoint=args.checkpoint)

