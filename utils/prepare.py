'''
Prepare Input Data
'''

import os
import glob

import numpy as np
import pandas as pd
import datetime
from bisect import bisect
from pathlib import Path
import platform

from config import *


def prepare_data(video_base, label_path, name_path):
    '''
    load all the videos and labels, processing the labels for the videos
    --------------------------------------------------------------------
    input:
        video_base : path of the videos
        label_path: path of labels for each video
        name_path: path of all the labels
    --------------------------------------------------------------------
    output:
        videos: path of videos
        labels_df: dataframe of the videos for each video
        all_labels_name: all the labels

    '''
    fmt_path = lambda x: windows_path(x) if platform.platform().startswith('Win') else x
    videos = sorted(fmt_path(glob.glob(os.path.join(video_base, '*.mp4'))))
    all_labels_df = pd.read_csv(label_path)
    all_labels_name = pd.read_csv(name_path, index_col='id')
    labels_df = all_labels_df.loc[
        all_labels_df['videoName'].isin(
            list(map(lambda s: s.split('/')[-1].split('.')[0], videos))
        )
    ].copy()
    # prepare labels
    t0 = datetime.datetime(1900, 1, 1)
    # convert end time to seconds
    labels_df.loc[:, 'Seconds'] = labels_df['End'].apply(
        # convertor
        lambda x: (datetime.datetime.strptime(
            x, '%M:%S' if len(x.split(':')) == 2 else '%H:%M:%S'
        ) - t0).seconds
    )
    labels_df.loc[:, 'Durations'] = labels_df.loc[:, 'Seconds'].diff()
    labels_df.loc[~(labels_df['Durations'] > 0), 'Durations'] = labels_df.loc[~(labels_df['Durations'] > 0), 'Seconds']
    labels_df.loc[:, 'SecondsAll'] = labels_df.loc[:, 'Durations'].cumsum()
    return videos, labels_df, all_labels_name


def get_class_weights(labels_df, a=0):
    label_cnts = labels_df[['PhaseName', 'Durations']].groupby('PhaseName').sum()['Durations']
    label_prob = label_cnts / label_cnts.sum()
    class_weights = (label_prob + a) / (1 + a * len(label_prob))
    return class_weights


def get_labels(labels_df, all_labels_name, num):
    all_labels = [labels_df.loc[bisect(labels_df['SecondsAll'], i), 'PhaseName'] for i in range(num)]
    return all_labels


def get_label_weights(all_labels, all_labels_name, class_weights, num):
    weights = list(map(lambda x: 1 / class_weights[all_labels_name.index[all_labels_name['labels'] == x][0] - 1], all_labels))
    return weights


def windows_path(path_list):
    temp = []
    for i in path_list:
        temp.append(Path(i).as_posix())
    return temp


def read(video_base, label_path, name_path, ind_end:int, ind_start=0) -> tuple[list, list]:
    '''
    Read image frames from videos.
    ---
    Input args:
        videos: Path of hernia surgical videos
        image_base: Path for saving image frames
        ind_end: Ending index of video sequences
        ind_start: Start index of video sequences
    ---
    Outputs:
        List of image paths and label paths
    '''
    # get 2 images and labels
    #dataset for test

    def sort_images(x):
        vid = int(x.split('_')[-1].split('/')[0])
        frame = int(x.split('.')[0].split('/')[-1].split('-')[0])
        return vid*7200 + frame

    videos, labels_df, names_df = prepare_data(video_base, label_path, name_path)
    fmt_path = lambda x: windows_path(x) if platform.platform().startswith('Win') else x
    image_paths = [sorted(
        fmt_path(glob.glob(hernia_image_pth + '/' + vid.split('/')[-1].split('.')[0] + '/*.png')), key=sort_images
    ) for vid in videos[ind_start:ind_end]]

    #image_paths = [list(filter(None, images)) for images in image_paths]
    #image_paths = [list(sorted(images, key=sort_images)) for images in image_paths]

    cnts = [len(imgs) for imgs in image_paths]
    elapsed = sum([list(map(lambda e: e / x, range(x))) for x in cnts], [])
    rsd = sum([list(map(lambda e: e / 60, range(x, 0, -1))) for x in cnts], [])
    image_paths = sum(image_paths, [])

    phases = get_labels(labels_df, names_df, len(image_paths))

    class_weights = get_class_weights(labels_df, a=alpha_smooth_class)
    weights = get_label_weights(phases, names_df, class_weights, len(image_paths))

    phases = [names_df[names_df['labels'] == x].index[0] for x in phases]

    return image_paths, phases, elapsed, rsd, class_weights, weights, cnts

