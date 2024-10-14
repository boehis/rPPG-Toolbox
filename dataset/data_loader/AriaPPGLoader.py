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
from scipy.interpolate import interp1d
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
        self.raw_imus = list()
        self.quaternions = list()
        super().__init__(name, data_path, config_data)


    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        raw_imu = np.load(self.raw_imus[index])
        quaternion = np.load(self.quaternions[index])

        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)
        label = np.float32(label)
        raw_imu = np.float32(raw_imu)
        quaternion = np.float32(quaternion)
        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id, raw_imu, quaternion

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


        # Read IMU
        raw_imu = self.read_imu(
            os.path.join(os.path.dirname(path),f"{cam_pos}_imu-left_{subject}_{task}_raw_imu.npy")
        )
        raw_imu = interp1d(
            np.arange(raw_imu.shape[0]),
            raw_imu, 
            kind='linear', axis=0, fill_value="extrapolate")(np.linspace(0,raw_imu.shape[0]-1, frames.shape[0]))

        quaternions = self.read_imu(
            os.path.join(os.path.dirname(path),f"{cam_pos}_imu-left_{subject}_{task}_quaternions.npy")
        )
        quaternions = interp1d(
            np.arange(quaternions.shape[0]),
            quaternions, 
            kind='linear', axis=0, fill_value="extrapolate")(np.linspace(0,quaternions.shape[0]-1, frames.shape[0]))

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            raw_imu_clips = [raw_imu[i * config_preprocess.CHUNK_LENGTH:(i + 1) * config_preprocess.CHUNK_LENGTH] for i in range(frames.shape[0] // config_preprocess.CHUNK_LENGTH)]
            quaternions_clips = [quaternions[i * config_preprocess.CHUNK_LENGTH:(i + 1) * config_preprocess.CHUNK_LENGTH] for i in range(frames.shape[0] // config_preprocess.CHUNK_LENGTH)]
        else:
            raw_imu_clips = np.array([raw_imu])
            quaternions_clips = np.array([quaternions])

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        logging.debug("Preprocess frames_clips with shape:", frames_clips.shape)
        logging.debug("Preprocess frames_clips with dtype:", frames_clips.dtype)

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, raw_imu_clips, quaternions_clips, index)
        file_list_dict[i] = input_name_list


    def save_multi_process(self, frames_clips, bvps_clips, raw_imu_clips, quaternions_clips, filename):
        """Save all the chunked data with multi-thread processing.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            raw_imu_path_name = self.cached_path + os.sep + "{0}_raw_imu{1}.npy".format(filename, str(count))
            quaternions_path_name = self.cached_path + os.sep + "{0}_quaternions{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            np.save(raw_imu_path_name, raw_imu_clips[i])
            np.save(quaternions_path_name, quaternions_clips[i])
            count += 1
        return input_path_name_list, label_path_name_list

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
        raw_imus = [input_file.replace("input", "raw_imu") for input_file in filtered_inputs]
        quaternions = [input_file.replace("input", "quaternions") for input_file in filtered_inputs]

        self.inputs = filtered_inputs
        self.labels = labels
        self.raw_imus = raw_imus
        self.quaternions = quaternions
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

    @staticmethod
    def read_imu(imu_file):
        """Reads a imu signal file."""
        return np.load(imu_file)