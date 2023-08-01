import logging
import os
import time
from tqdm import tqdm
import torch
import numpy as np
import math
from json_io import predictions2json
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2
import json


def test(dataloader,model, model_name, NMS_window=30, NMS_threshold=0.5, save_predictions=False):
    #split = '_'.join(dataloader.dataset.split)
    #output_results = os.path.join("models", model_name, f"results_spotting_{split}.zip")
    #output_folder = f"outputs_{split}"
    spotting_predictions = list()

    model.eval()

    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, size) in t:
            feat_half1 = feat_half1.cpu().squeeze(0)

            # Compute the output for batches of frames
            BS = 256
            timestamp_long_half_1 = []
            for b in range(int(np.ceil(len(feat_half1) / BS))):
                start_frame = BS * b
                end_frame = BS * (b + 1) if BS * \
                                            (b + 1) < len(feat_half1) else len(feat_half1) - 1
                feat = feat_half1[start_frame:end_frame].cpu()
                output = model(feat).cpu().detach().numpy()
                timestamp_long_half_1.append(output)
            timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

            timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
            spotting_predictions.append(timestamp_long_half_1)


        def get_spot_from_NMS(Input, window, thresh=0.0):

            detections_tmp = np.copy(Input)
            # res = np.empty(np.size(Input), dtype=bool)
            indexes = []
            MaxValues = []
            while (np.max(detections_tmp) >= thresh):
                # Get the max remaining index and value
                max_value = np.max(detections_tmp)
                max_index = np.argmax(detections_tmp)
                MaxValues.append(max_value)
                indexes.append(max_index)
                # detections_NMS[max_index,i] = max_value

                nms_from = int(np.maximum(-(window / 2) + max_index, 0))
                nms_to = int(np.minimum(max_index + int(window / 2), len(detections_tmp)))
                detections_tmp[nms_from:nms_to] = -1

            return np.transpose([indexes, MaxValues])

        framerate = dataloader.dataset.framerate
        get_spot = get_spot_from_NMS

        json_data = dict()
        json_data["predictions"] = list()

        for l in range(dataloader.dataset.num_classes):
            spots = get_spot(
                timestamp_long_half_1[:, l], window=NMS_window * framerate, thresh=NMS_threshold)
            for spot in spots:
                # print("spot", int(spot[0]), spot[1], spot)
                frame_index = int(spot[0])
                confidence = spot[1]
                # confidence = predictions_half_1[frame_index, l]

                seconds = int((frame_index // framerate) % 60)
                minutes = int((frame_index // framerate) // 60)

                prediction_data = dict()
                prediction_data["gameTime"] = "1 - " + str(minutes) + ":" + str(seconds)
                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                prediction_data["position"] = str(int((frame_index / framerate) * 1000))
                prediction_data["half"] = str(1)
                prediction_data["confidence"] = str(confidence)
                json_data["predictions"].append(prediction_data)

    # Save the predictions to the json format
    with open("inference/outputs/Predictions-v2.json", 'w') as output_file:
        json.dump(json_data, output_file, indent=4)
