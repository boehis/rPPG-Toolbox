"""The dataloader for CameraBP datasets.
"""
import glob
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import pandas as pd


class CameraBPLoader(BaseLoader):
    """The data loader for the CameraBP dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an CameraBP dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "CameraBP" for below dataset structure:
                -----------------
                    CameraBP
                    ├── 001
                    │   ├── 001_basler_face.mp4
                    │   ├── 001_basler_hand.mp4
                    │   └── 001_biopac.csv
                    ├── 002
                    │   ├── 002_basler_face.mp4
                    │   ├── 002_basler_hand.mp4
                    │   └── 002_biopac.csv
                    |...
                    ├── xxx
                    │   ├── xxx_basler_face.mp4
                    │   ├── xxx_basler_hand.mp4
                    │   └── xxx_biopac.csv

                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For CameraBP dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "[0-9][0-9][0-9]")  # Matches exactly three digits
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.split(data_dir)[-1].split('_')[0]
            index = os.path.split(data_dir)[-1].split('.')[0]

            # Split Phases: Read the CSV file to extract the phase start time
            df = pd.read_csv(os.path.join(data_dir, f"{subject}_biopac.csv"))

            experiment_start_time = df['timestamps'].iloc[0]
            experiment_end_time = df['timestamps'].iloc[-1]
            phase_start_times = (df[df['events'] == 1]['timestamps'] - experiment_start_time).values
            # Todo: Depending on the dataset, the first phase migh or might not have a start. Check and maybe add 0 at start. 
            phase_names = [
                "rest", "hand grip", "rest", "mental arithmetic", "rest", "hand grip", "rest", 
                "deep breathing", "rest", "hand grip", "rest", "cold pressor", "rest", "valsalva maneuver", "rest"
            ]

            # Ensure the length of phase_start_times matches the number of expected phases
            if len(phase_start_times) != len(phase_names):
                raise ValueError("The number of detected phases does not match the expected number of phases.")


            for i, start_time in enumerate(phase_start_times):
                phase_name = phase_names[i]
                end_time = phase_start_times[i+1] if i < len(phase_start_times)-1 else (experiment_end_time-experiment_start_time) 

                dirs.append({
                    "index": f"{subject}_{phase_name}_{i}",
                    "path": data_dir,
                    "subject": subject,
                    "phase": phase_name,
                    "phase_index": i,
                    "start_time_s": start_time.round(2), # Round to 100th of a second matching the video frame rate of 100fps
                    "end_time_s": end_time.round(2) # Round to 100th of a second matching the video frame rate of 100fps
                })
                print("index", f"{subject}_{phase_name}_{i}", "duration_m:", (end_time.round(2)-start_time.round(2))/60)
                
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""
        if begin == 0 and end == 1:
            return data_dirs
    
        subjects = sorted(list(set(d['subject'] for d in data_dirs)))
        nr_subjects = len(subjects)
        begin_idx = int(nr_subjects * begin)
        end_idx = int(nr_subjects * end)

        selected_subjects = subjects[begin_idx:end_idx]
        selected_data_dirs = [d for d in data_dirs if d['subject'] in selected_subjects]
        return selected_data_dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        print("i", i)
        saved_filename = data_dirs[i]['index']

        print("load_f")
        frames = self.read_video(
            video_file=     os.path.join(data_dirs[i]['path'], f"{data_dirs[i]['subject']}_basler_face.mp4"),
            start_time_s=   data_dirs[i]['start_time_s'],
            #end_time_s=     data_dirs[i]['end_time_s']
            end_time_s=     data_dirs[i]['start_time_s']+1
            )
        print(frames.shape)

        # Read Labels
        bvps = self.read_wave(
            bvp_file=       os.path.join(data_dirs[i]['path'], f"{data_dirs[i]['subject']}_biopac.csv"),
            start_time_s=   data_dirs[i]['start_time_s'],
            end_time_s=     data_dirs[i]['end_time_s']
            )
        print(bvps.shape)

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        print("prep")
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        print("frames_clips", frames_clips.shape)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list
        print(file_list_dict)

    @staticmethod
    def read_video(video_file, start_time_s, end_time_s):
        """Reads a video file, returns frames(T, H, W, 3) """
        cap = cv2.VideoCapture(video_file)

        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        FPS = cap.get(cv2.CAP_PROP_FPS)
        assert FPS == 100, "The FPS of the video is not 100. Please check the video file."

        T = int((end_time_s - start_time_s) * FPS)

        video = np.zeros((T, H, W, 3), dtype=np.uint8)
        #if memory is a problem, use memmap to store the video
        # video = np.memmap('video.dat', dtype='uint8', mode='w+', shape=(T, H, W, 3))

        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_s * 1000)
        for frame_nr in range(T):
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Error reading frame {frame_nr} from video {video_file}")
            video[frame_nr] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return video

    @staticmethod
    def read_wave(bvp_file, start_time_s, end_time_s):
        """Reads a bvp signal file."""
        df = pd.read_csv(bvp_file)

        start_timestamp = df['timestamps'][0]

        block_df = df[(df['timestamps'] > start_timestamp + start_time_s) & (df['timestamps'] < start_timestamp + end_time_s)]

        return block_df['PPG_ear'].values