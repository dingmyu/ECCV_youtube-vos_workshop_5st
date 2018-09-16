#!/bin/bash
srun --mpi=pmi2 -p Segmentation1080 -n1 --gres=gpu:1 --ntasks-per-node=1  python -u davis_test_muti_label.py --weights /mnt/lustre/sunpeng/Research/video-seg-workshop/models/deeplabv3_models/final_2s_seg_youtube_online_two/exp/drivable/youtube_2s_2pairs_4epoch_fix_final_mask_0809/model/train_epoch_2.pth --output out_test --gpu_num 1 --gpu 1
