#!/bin/sh
exp=youtube_2s_0816_add_davis_4epoch_biglr2
EXP_DIR=exp/drivable/$exp
mkdir -p ${EXP_DIR}/model
now=$(date +"%Y%m%d_%H%M%S")
#cp train.sh train.py ${EXP_DIR}
part=Segmentation1080
#part=Test
numGPU=16 #8
nodeGPU=8
# coco 36
# voc 21

GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $part --gres=gpu:$nodeGPU -n$numGPU --ntasks-per-node=$nodeGPU --job-name=${exp} \
python -u train.py \
  --data_root= \
  --val_list=/mnt/lustre/sunpeng/Research/image-base-workshop/anue/lists/val_list.txt \
  --layers=101 \
  --backbone=resnet \
  --net_type=0 \
  --port=12345 \
  --syncbn=1 \
  --classes=2 \
  --crop_h=433 \
  --crop_w=433 \
  --zoom_factor=1 \
  --base_lr=1e-4 \
  --epochs=2 \
  --start_epoch=1 \
  --batch_size=1 \
  --bn_group=$numGPU \
  --save_step=1 \
  --save_path=${EXP_DIR}/model \
  --evaluate=0 \
  --weight=/mnt/lustre/sunpeng/Research/video-seg-workshop/models/deeplabv3_models/final_patch_seg_youtube_online_fix_patch/exp/drivable/youtube_rgb_fix_patch_2epoch/model/train_epoch_2.pth \
  --weight_flow=/mnt/lustre/sunpeng/Research/video-seg-workshop/models/deeplabv3_models/final_flow_patch_seg_youtube_online/exp/drivable/youtube_slr_from_rgb/model/train_epoch_2.pth \
  --ignore_label 255 \
  --workers 2 \
  --dataset_name=youtube_and_davis \
  2>&1 | tee ${EXP_DIR}/model/train-$now.log
