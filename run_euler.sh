#!/bin/bash

cd "$(dirname "$0")"

module load stack/2024-06 python_cuda/3.9.18
# UNSUPERVISED
sbatch --output=./configs/infer_configs/aria_ppg/UNSUPERVISED/lp1.out    --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UNSUPERVISED/lp1.yaml"    --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G
sbatch --output=./configs/infer_configs/aria_ppg/UNSUPERVISED/lp3.out    --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UNSUPERVISED/lp3.yaml"    --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G
sbatch --output=./configs/infer_configs/aria_ppg/UNSUPERVISED/lp5.out    --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UNSUPERVISED/lp5.yaml"    --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G
sbatch --output=./configs/infer_configs/aria_ppg/UNSUPERVISED/median.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UNSUPERVISED/median.yaml" --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G
sbatch --output=./configs/infer_configs/aria_ppg/UNSUPERVISED/none.out   --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UNSUPERVISED/none.yaml"   --time=100 --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G

# PURE_AriaPPG_PHYSNET_BASIC
sbatch --output=./configs/infer_configs/aria_ppg/PURE_AriaPPG_PHYSNET_BASIC/pxt1stat.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/PURE_AriaPPG_PHYSNET_BASIC/pxt1stat.yaml" --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=20
sbatch --output=./configs/infer_configs/aria_ppg/PURE_AriaPPG_PHYSNET_BASIC/pxt1stat_shifted.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/PURE_AriaPPG_PHYSNET_BASIC/pxt1stat_shifted.yaml" --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=20
sbatch --output=./configs/infer_configs/aria_ppg/PURE_AriaPPG_PHYSNET_BASIC/pxt1.out     --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/PURE_AriaPPG_PHYSNET_BASIC/pxt1.yaml"     --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=40
sbatch --output=./configs/infer_configs/aria_ppg/PURE_AriaPPG_PHYSNET_BASIC/pxtx.out     --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/PURE_AriaPPG_PHYSNET_BASIC/pxtx.yaml"     --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=120

# rPPG_AriaPPG_PHYSNET_BASIC
sbatch --output=./configs/infer_configs/aria_ppg/UBFC-rPPG_AriaPPG_PHYSNET_BASIC/pxt1stat.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UBFC-rPPG_AriaPPG_PHYSNET_BASIC/pxt1stat.yaml" --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=20
sbatch --output=./configs/infer_configs/aria_ppg/UBFC-rPPG_AriaPPG_PHYSNET_BASIC/pxt1stat_shifted.out --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UBFC-rPPG_AriaPPG_PHYSNET_BASIC/pxt1stat_shifted.yaml" --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=20
sbatch --output=./configs/infer_configs/aria_ppg/UBFC-rPPG_AriaPPG_PHYSNET_BASIC/pxt1.out     --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UBFC-rPPG_AriaPPG_PHYSNET_BASIC/pxt1.yaml"     --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=40
sbatch --output=./configs/infer_configs/aria_ppg/UBFC-rPPG_AriaPPG_PHYSNET_BASIC/pxtx.out     --open-mode=truncate --wrap="python ./main.py --config ./configs/infer_configs/aria_ppg/UBFC-rPPG_AriaPPG_PHYSNET_BASIC/pxtx.yaml"     --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=120



sbatch --output=./configs/train_configs/aria_ppg/PHYSNET/pxt1stat_raw.out --open-mode=truncate --wrap="python ./main.py --config ./configs/train_configs/aria_ppg/PHYSNET/pxt1stat_raw.yaml" --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=40
sbatch --output=./configs/train_configs/aria_ppg/PHYSNET/pxt1stat_diffnormalized.out --open-mode=truncate --wrap="python ./main.py --config ./configs/train_configs/aria_ppg/PHYSNET/pxt1stat_diffnormalized.yaml" --nodes=1 --cpus-per-task=2 --mem-per-cpu=5G --gpus=1 --time=40
