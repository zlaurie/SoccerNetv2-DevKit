from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time
import ffmpy

from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import SoccerNetDownloader
from Features.VideoFeatureExtractor import VideoFeatureExtractor, PCAReducer


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy",framerate=2, chunk_size=240, receptive_field=80, window_size = 15):
        self.path = path
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate = framerate
        self.num_classes = 17
        self.num_detections =15
        self.window_size_frame = window_size*framerate


        #Changing video format to 
        #ff = ffmpy.FFmpeg(inputs={self.path: ""},outputs={"inference/outputs/videoLQ.mkv": '-y -r 25 -vf scale=-1:224 -max_muxing_queue_size 9999'})
        #print(ff.cmd)
        #ff.run()

        print("Initializing feature extractor")
        #myFeatureExtractor = VideoFeatureExtractor(feature="ResNET",back_end="TF2",transform="crop",grabber="opencv",FPS=self.framerate)

        print("Extracting frames")
        #myFeatureExtractor.extractFeatures(path_video_input="inference/outputs/videoLQ.mkv",path_features_output="inference/outputs/features.npy",overwrite=True)

        print("Initializing PCA reducer")
        #myPCAReducer = PCAReducer(pca_file="inference/Features/pca_512_TF2.pkl",scaler_file="inference/Features/average_512_TF2.pkl")

        print("Reducing with PCA")
        # myPCAReducer.reduceFeatures(input_features="inference/outputs/features.npy",output_features="inference/outputs/features_PCA.npy",overwrite=True)

        print("Reducing with PCA")
        #myPCAReducer.reduceFeatures_nopca(input_features="inference/outputs/features.npy",output_features="inference/outputs/features_avg.npy",overwrite=True)

    def __getitem__(self, index):
        # Load features
        feat_half1 = np.load(os.path.join("inference/outputs/features_avg.npy"))
        feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])

        print("Shape half 1: ", feat_half1.shape)
        size = feat_half1.shape[0]

        def feats2clip(feats, stride, clip_length, off=0):
            idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
            idxs = []
            for i in torch.arange(-off, clip_length - off):
                # for i in torch.arange(0, clip_length):
                idxs.append(idx + i)
            idx = torch.stack(idxs, dim=1)

            idx = idx.clamp(0, feats.shape[0] - 1)
            # Not replicate last, but take the clip closest to the end of the video
            # idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length
            return feats[idx, ...]

        #feat_half1 = feats2clip(torch.from_numpy(feat_half1),stride=1, off=int(self.chunk_size / 2),clip_length=self.chunk_size)
        feat_half1 = feats2clip(torch.from_numpy(feat_half1),stride=1, off=int(self.window_size_frame / 2),clip_length=self.window_size_frame)

        return feat_half1, size

    def __len__(self):
        return 1