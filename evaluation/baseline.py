import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import chirp, find_peaks
from joblib import Parallel, delayed
tqdm.pandas()

from post_process import calculate_metric_per_video
from post_process import _calculate_fft_hr
from post_process import _calculate_peak_hr

# Paths for unsupervised data
UNSUPERVISED_ROOT_PATH = '/cluster/scratch/boehis/runs/infer_configs/UNSUPERVISED/'
UNSUPERVISED_PATH = os.path.join(UNSUPERVISED_ROOT_PATH, 'median', 'len128', 'res128', 'AriaPPG_SizeW128_SizeH128_ClipLength128_DataTypeRaw_DataAugNone_LabelTypeRaw_Crop_faceFalse_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_unsupervised', 'saved_outputs')


def main():
    if os.path.exists(UNSUPERVISED_PATH):
        data_frame = pd.read_csv(os.path.join(UNSUPERVISED_PATH, 'POS_all_results.csv'))
        data_frame['PID'] = data_frame['index'].str[4:8]  # Assuming PID is in this position

        data_frame['label'] = data_frame['label'].apply(eval).apply(np.array)
        baseline = []
        for pid in data_frame['PID'].unique():
            train = data_frame[data_frame['PID'] != pid]
            test = data_frame[data_frame['PID'] == pid].copy()

            test['prediction'] = [train['label'].mean()] * len(test)
            baseline.append(test)
        baseline = pd.concat(baseline)
        baseline['label'] = baseline['label'].apply(list)
        baseline['prediction'] = baseline['prediction'].apply(list)
        # Save the result
        
        output_path = os.path.join(UNSUPERVISED_PATH, 'Baseline_all_results.csv')
        baseline.to_csv(output_path, index=False)
        print(f"Baseline metrics saved to {output_path}")
    else:
        print(f"Path does not exist: {UNSUPERVISED_PATH}")

# Run the main function
if __name__ == "__main__":
    main()