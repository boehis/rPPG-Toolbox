"""The dataloader for the AriaPPG dataset.
"""
import glob
import os
import re
import math
import multiprocessing
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd

import logging
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')


class AriaPPGLoader(BaseLoader):
    """The data loader for the AriaPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an AriaPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "AriaPPG" for below dataset structure:
                -----------------
                AriaPPG/
                    ├── P001
                    │   ├── T1
                    │   │   ├── e4_BVP_P001_T1.csv
                    │   │   ├── e4_EDA_P001_T1.csv
                    │   │   ├── ...
                    │   │   ├── head_camera-rgb_P001_T1.mkv
                    │   │   ├── head_imu-left_P001_T1.csv
                    │   │   ├── ...
                    │   │   ├── stat_camera-rgb_P001_T1.mkv
                    │   │   ├── stat_imu-left_P001_T1.csv
                    │   │   ├── ...
                    │   ├── ...
                    ├── ...
                    ├── PNNN
                    │   ├── T1
                    │   │   ├── ...
                    │   ├── ...
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For AriaPPG dataset)."""
        data_dirs = glob.glob(os.path.join(data_path,"P[0-9][0-9][0-9]", "T[0-9]","*_camera-rgb_*.npy"))
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        dirs = []
        for data_dir in data_dirs:
            match = re.search('^.*(head|stat)_camera-rgb_(P\d{3})_(T.*).npy$', data_dir)
            cam_pos = match.group(1)
            subject = match.group(2)
            task = match.group(3)
            dirs.append({
                "index": cam_pos+subject+task,
                "cam_pos": cam_pos,
                "subject": subject,
                "task": task,
                "path": data_dir,
            })
        
        return dirs

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses all the raw data based on split.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)  # partition dataset 
        # send data directories to be processed

        # Get the number of CPUs allocated to your job
        num_cores = int(os.getenv('SLURM_CPUS_PER_TASK', default=1))
        logging.debug("Num avail cores:", num_cores)
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess, multi_process_quota=num_cores)
        self.build_file_list(file_list_dict)  # build file list
        self.load_preprocessed_data()  # load all data and corresponding labels (sorted for consistency)
        logging.debug("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

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
        index = data_dirs[i]['index']
        cam_pos = data_dirs[i]['cam_pos']
        subject = data_dirs[i]['subject']
        task = data_dirs[i]['task']
        path = data_dirs[i]['path']
        

        # Read Frames
        frames = self.read_video(path)

        logging.debug("Read frames with shape:", frames.shape)
        logging.debug("Read frames with dtype:", frames.dtype)

        # Read Labels
        bvps = self.read_wave(
                os.path.join(os.path.dirname(path),f"e4_BVP_{subject}_{task}.npy"))

        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])

        logging.debug("Read Labels with shape:", bvps.shape)

        
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        logging.debug("Preprocess frames_clips with shape:", frames_clips.shape)
        logging.debug("Preprocess frames_clips with dtype:", frames_clips.dtype)

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, index)
        file_list_dict[i] = input_name_list

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
            print(input)
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]
            subject = input_name[4:8]
            input_idx = int(input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[1][5:])
            if self.filtering.USE_EXCLUSION_LIST and input_name in self.filtering.EXCLUSION_LIST :
                # Skip loading the input as it's in the exclusion list
                continue
            if self.filtering.SELECT_TASKS and not any(task in input_name for task in self.filtering.TASK_LIST):
                # Skip loading the input as it's not in the task list
                continue
            if self.filtering.SELECT_INDICES and not input_idx in self.filtering.INDEX_LIST:
                # Skip loading the input as it's not in the index list
                continue
            if self.filtering.SELECT_CAM and not input_name[:4] in self.filtering.CAM_LIST:
                # Skip loading the input as it's not in the camera list
                continue
            if self.filtering.LOSO_CV and not(subject in self.filtering.SUBJECT_LIST):
                # Skip subject if it's LOSO subject
                continue
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

        # print(filtered_inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        return np.load(video_file).astype(np.float32)/255

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        return np.load(bvp_file)