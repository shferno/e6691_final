'''
Datasets and loaders. 

- Construct datasets from videos
- Sampling & augmentations
- Dataloaders
'''


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import cv2
from PIL import Image
from bisect import bisect

import os
import glob
from utils.prepare import windows_path
import platform
import numpy as np


class SVRCDataset(Dataset):
    '''
    Input data structure for SVRCNet,
    '''
    def __init__(self, image_path: list, labels: tuple, transform):
        self.image_path = image_path
        self.image_class = labels[0]
        self.elapsed = labels[1]
        self.rsd = labels[2]
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item): #can add more rules to pick data
        #img = Image.open(self.image_path[item])
        img = cv2.cvtColor(cv2.imread(self.image_path[item]), cv2.COLOR_BGR2RGB)
        #if self.image_class is not None:
        label = (self.image_class[item], self.rsd[item]) if self.image_class is not None else None
        img = self.transform(img.astype(float) / 255)
        time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * self.elapsed[item]
        img = torch.cat((img, time_stamp.float()), dim=0)
        label = torch.tensor(label)

        return {'feature': img, 'label': label} # if self.image_class is not None else {'feature': img}


class HerniaDataset(Dataset):
    '''
    Dataset for Hernia. 
    '''

    def __init__(self, videos, labels=None, names=None, transforms=None) -> None:
        '''
        Initialize the dataset. 
        ----
        Parameters
        - videos: list -> a list of all videos
        - labels: pd.DataFrame -> a dataframe of all phases
        '''

        self.videos = videos
        # VideoCapture objects
        #self.caps = [cv2.VideoCapture(videos[0])]
        # accumulated frame counts
        self.cnts = [int(cv2.VideoCapture(videos[0]).get(cv2.CAP_PROP_FRAME_COUNT))]
        for vid in videos[1:]:
            self.cnts.append(self.cnts[-1] + int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FRAME_COUNT)))

        self.labels = labels
        self.names = names
        self.transform = transforms


    def __len__(self):
        ''' number of all images '''
        return self.cnts[-1]


    def __getitem__(self, index):
        ''' get an item '''
        # get video_id with binary search
        vid_id = bisect(self.cnts, index)
        cap = cv2.VideoCapture(self.videos[vid_id])
        # relative frame_id within the video
        frame_id = index if vid_id == 0 else index - self.cnts[vid_id - 1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = cap.read()
        while not ret:
            print('failed on {}:{}'.format(self.videos[vid_id], frame_id), end='\r')
            ret, img = cap.read()
            if ret:
                print('\nsucceed')
        cap.release()
        img = self.transform(torch.FloatTensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255).permute(2, 0, 1))
        time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_id
        img = torch.cat((img, time_stamp.float()), dim=0)

        # get label
        phase = self.names[
            # select label_id of phase name
            self.names['labels'] == self.labels.loc[
                # select phase name of frame_id
                bisect(self.labels['SecondsAll'], frame_id), 'PhaseName'
            ]
        ].index[0] if self.labels is not None else None

        # assemble
        rsd = self.cnts[vid_id] - index
        label = np.array([phase, 1, 1, 1, frame_id, rsd])

        return img, label


class DatasetNoLabel(Dataset):
    """
    Dataset for Cataract.
    Dataset for folders with sampled png images from videos
    """
    def __init__(self, datafolders, img_transform=None, max_len=20, fps=2.5):
        super(DatasetNoLabel).__init__()
        self.datafolders = datafolders
        self.img_transform = img_transform
        self.max_len = max_len*fps*60.0
        self.frame2min = 1/(fps*60.0)
        # glob files
        self.surgery_length = {}
        img_files = []
        for d in datafolders:
            files = sorted(glob.glob(os.path.join(d, '*.png')))
            fmt_path = lambda x: windows_path(x) if platform.platform().startswith('Win') else x
            img_files += fmt_path(files[:-2])
            patientID, frame = self._name2id(files[-1])
            self.surgery_length[patientID] = float(frame)*self.frame2min
        img_files = sorted(img_files)
        assert len(img_files) > 0, 'no png images found in {0}'.format(datafolders)

        self.img_files = img_files
        self.nitems = len(self.img_files)

    def __getitem__(self, index):
        # load image
        ret = cv2.imread(self.img_files[index])
        if ret is None:
            print('weird: ', self.img_files[index])
        img = cv2.cvtColor(cv2.imread(self.img_files[index]), cv2.COLOR_BGR2RGB)

        patientID, frame_number = self._name2id(self.img_files[index])
        elapsed_time = frame_number*self.frame2min
        rsd = self.surgery_length[patientID] - elapsed_time
        if self.img_transform is not None:
            img = self.img_transform(img)
        time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_number / self.max_len
        # append to image as additional channel
        img = torch.cat((img, time_stamp.float()), dim=0)
        return img, elapsed_time, frame_number, rsd

    def __len__(self):
        return self.nitems

    def _name2id(self, filename):
        *patientID, frame = os.path.splitext(os.path.basename(filename))[0].split('_')
        patientID = '_'.join(patientID)
        return patientID, int(frame)-1


class CataDataset(DatasetNoLabel):
    """
    Dataset for Cataract.
    Dataset for folders with sampled png images from videos
    """
    def __init__(self, datafolders, label_files, img_transform=None, max_len=20, fps=2.5):
        super().__init__(datafolders, img_transform, max_len, fps)
        assert len(label_files) == len(datafolders), 'not the same number of data and label files'
        self.label_files = {}
        for f in label_files:
            patientID = os.path.splitext(os.path.basename(f))[0]
            self.label_files[patientID] = np.genfromtxt(f, delimiter=',', skip_header=1)[:, 1:]

    def __getitem__(self, index):
        img, elapsed_time, frame_number, rsd = super().__getitem__(index)

        # load label
        patientID, frame_number = self._name2id(self.img_files[index])
        try:
            label = self.label_files[patientID.split('_')[-1]][frame_number]
            self.last_label = label
        except:
            label = self.last_label
        # label = self.label_files[patientID.split('_')[-1]][frame_number]
        return img, label

