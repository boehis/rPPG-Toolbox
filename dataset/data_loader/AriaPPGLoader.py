"""The dataloader for the AriaPPG dataset.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd

class AriaPPGLoader(BaseLoader):
    """The data loader for the AriaPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an AriaPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "AriaPPG" for below dataset structure:
                -----------------
                     AriaPPG/
                     |   |-- P001/
                     |       |-- vid_P001_T1.mkv
                     |       |-- vid_P001_T2.mkv
                     |       |-- vid_P001_T3.mkv
                     |       |...
                     |       |-- bvp_P001_T1.csv
                     |       |-- bvp_P001_T2.csv
                     |       |-- bvp_P001_T3.csv
                     |   |-- P002/
                     |       |-- vid_P002_T1.mkv
                     |       |-- vid_P002_T2.mkv
                     |       |-- vid_P002_T3.mkv
                     |       |...
                     |       |-- bvp_P002_T1.csv
                     |       |-- bvp_P002_T2.csv
                     |       |-- bvp_P002_T3.csv
                     |...
                     |   |-- PNNN/
                     |       |-- vid_Pnnn_T1.mkv
                     |       |-- vid_Pnnn_T2.mkv
                     |       |-- vid_Pnnn_T3.mkv
                     |       |...
                     |       |-- bvp_Pnnn_T1.csv
                     |       |-- bvp_Pnnn_T2.csv
                     |       |-- bvp_Pnnn_T3.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For AriaPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "P[0-9][0-9][0-9]" + os.sep + "*.mkv")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{
            "index": re.search('vid_(.*).mkv', data_dir).group(1),
            "subject": re.search('vid_(.*)_(.*).mkv', data_dir).group(1),
            "path": data_dir,
            } for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        
        subjects = sorted(list(set(d['subject'] for d in data_dirs)))
        nr_subjects = len(subjects)
        begin_idx = int(nr_subjects * begin)
        end_idx = int(nr_subjects * end)

        selected_subjects = subjects[begin_idx:end_idx]
        selected_data_dirs = [d for d in data_dirs if d['subject'] in selected_subjects]
        return selected_data_dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        print("Read Frames")
        # Read Frames
        frames = self.read_video(
            os.path.join(data_dirs[i]['path']))

        print("Read frames with shape:", frames.shape)
        print("Read frames with dtype:", frames.dtype)

        print("Read Labels")
        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            print("using USE_PSUEDO_PPG_LABEL")
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(os.path.dirname(data_dirs[i]['path']),f"bvp_{saved_filename}.csv"))

        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])

        print("Read Labels with shape:", bvps.shape)

        
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        frames_clips = frames_clips.astype(np.uint8)

        print("Preprocess frames_clips with shape:", frames_clips.shape)
        print("Preprocess frames_clips with dtype:", frames_clips.dtype)

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list
        print(f"file_list_dict[{i}]:",file_list_dict[i])

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []

        for input in base_inputs:
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]
            if self.filtering.USE_EXCLUSION_LIST and input_name in self.filtering.EXCLUSION_LIST :
                # Skip loading the input as it's in the exclusion list
                continue
            if self.filtering.SELECT_TASKS and not any(task in input_name for task in self.filtering.TASK_LIST):
                # Skip loading the input as it's not in the task list
                continue
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_file}")

        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        FPS = cap.get(cv2.CAP_PROP_FPS)
        assert FPS == 30, f"The FPS of the video is {FPS}, not 30. Please check the video file."


        video = np.zeros((T, H, W, 3), dtype=np.uint8)
        #if memory is a problem, use memmap to store the video
        #dat_file = video_file + '.dat'
        #if os.path.exists(dat_file):
        #    print("Read existing .dat file")
        #    video = np.memmap(dat_file, dtype='uint8', mode='r+', shape=(T, H, W, 3))
        #    cap.release()
        #    return video
        #    
        #video = np.memmap(dat_file, dtype='uint8', mode='w+', shape=(T, H, W, 3))

        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        for frame_nr in range(T):
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Error reading frame {frame_nr} from video {video_file}")
            video[frame_nr] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return video

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        return pd.read_csv(bvp_file, header=None)[0].values