#!/bin/bash

cd "$(dirname "$0")"

module load stack/2024-06 python_cuda/3.9.18
sbatch --output=./AriaPPG_UNSUPERVISED_lp1.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/AriaPPG_UNSUPERVISED_lp1.yaml" --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G
sbatch --output=./AriaPPG_UNSUPERVISED_lp3.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/AriaPPG_UNSUPERVISED_lp3.yaml" --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G
sbatch --output=./AriaPPG_UNSUPERVISED_lp5.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/AriaPPG_UNSUPERVISED_lp5.yaml" --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G
sbatch --output=./AriaPPG_UNSUPERVISED_med.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/AriaPPG_UNSUPERVISED_med.yaml" --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G
sbatch --output=./AriaPPG_UNSUPERVISED_none.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/AriaPPG_UNSUPERVISED_none.yaml" --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G