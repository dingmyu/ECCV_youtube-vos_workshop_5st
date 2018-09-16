from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import os
from os.path import *
import cv2
import json
from glob import glob
import pycocotools.mask as maskUtils
from PIL import Image
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)
cudnn.benchmark = True
height, width = (256, 128)
model = models.create('resnet50', num_features=1024,
                          dropout=0, num_classes=128)
model = model.cuda()
checkpoint = load_checkpoint(osp.join('/mnt/lustre/dingmingyu/workspace/experiments/youtube/open-reid/examples/logs/triplet-loss/finetune','model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])
normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

test_transformer = T.Compose([
    T.RectScale(height, width),
    T.ToTensor(),
    normalizer,
])
model.eval()


prob_dir = '/mnt/lustre/dingmingyu/workspace/experiments/youtube/REID_0.5/0.5Annotations/'
JPEG_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/final_test_set/test_all_frames/JPEGImages/"
Annotations_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/final_test_set/test_all_frames/mat_label/"
flow_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/final_test_set/test_all_frames/flows/flow_mean/"
inverse_flow_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/final_test_set/test_all_frames/flows/flow_inverse_mean/"
json_dir = '/mnt/lustre/sunpeng/Research/pytorch-object-detection/examples/mask-rcnn/coco/ResNet101-FPN-Mask-16/results_test_dir/results/'

def gen_bbox(label, num):
    y, x = np.where(label == num)
    w = x.max()- x.min()
    h = y.max() - y.min()
    x1,x2,y1,y2 =  max(x.min()- int(0.05*w),0),min(x.max()+int(0.05*w),label.shape[1]-1), max(y.min()-int(0.05*h),0), min(y.max()+int(0.05*h),label.shape[0]-1)
    return [x1,y1,x2,y2]

def IoU(bbox1, bbox2):
    s1 = max(0, (min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + 1)) * max(0, (min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + 1))
    s2 = (max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1] + 1)) * (max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0] + 1))
    return float(s1) / float(s2)

def seg_IoU(seg1, seg2):
    inter = (seg1*seg2).sum()
    union = (seg1 + seg2 - seg1 * seg2).sum()
    return float(inter) / float(union)



video_dir_all = os.listdir(prob_dir)
video_dir_all.sort()
#if not os.path.exist(sys.argv[3] + sys.argv[4] + 'object_json'):
#    os.mkdir(sys.argv[3] + sys.argv[4] + 'object_json')
#os.mkdir(sys.argv[3] + sys.argv[4] + sys.argv[1])
for video_dir in video_dir_all[int(sys.argv[1]):int(sys.argv[2])]:
    print(video_dir)
    video_dir = video_dir.strip()
    frame_dir = os.path.join(JPEG_dir, video_dir)
    frame_nums_flow = len(os.listdir(frame_dir))
    label_dir = os.path.join(Annotations_dir, video_dir)
    flow_dir1 = os.path.join(flow_dir, video_dir)
    flow_dir2 = os.path.join(inverse_flow_dir, video_dir)
    frame_fr_dir = frame_dir
    # cache_dir = os.path.join(cache_dir, video_dir)
    # val_examples = sorted(os.listdir(os.path.join(val_example, video_dir)))
    gt_labels = sorted(os.listdir(label_dir))
    det_result = json.loads(open(json_dir + video_dir + '.txt').read())

    if len(gt_labels) == 1:
        gt_labels_tuple = [(gt_labels[0], -100)]
    else:
        gt_labels_b = gt_labels[1:]
        gt_labels_b.append(-100)
        gt_labels_tuple = zip(gt_labels, gt_labels_b)
    head_num = int(gt_labels_tuple[0][0][:5])
    search_instance = []
    pic_list= []
    pic_name_list = os.listdir(prob_dir + video_dir)
    pic_name_list.sort()
    instance_num = 0
    for index, probname in enumerate(pic_name_list):
        pic = cv2.imread(prob_dir + video_dir + '/' + probname)
        pic_list.append(pic)
    for (begin_label, end_label) in gt_labels_tuple:
        img_list = sorted( glob( join(frame_fr_dir, '*.jpg') ) )
        start_path = join(frame_fr_dir, begin_label.replace(".png",".jpg"))
        start_index = img_list.index(start_path)
        img_list_from_begin_label = img_list[start_index:]
        img_list_excpet_0 = img_list[start_index + 1:]
        #print(img_list_from_begin_label[0], os.path.join(label_dir, begin_label))
        wrapprob1 = prob_dir + video_dir + '/%05d.png' % (int(img_list_from_begin_label[1][-9:-4]) - head_num)
        wrapprob2 = prob_dir + video_dir + '/%05d.png' % (int(img_list_from_begin_label[2][-9:-4]) - head_num)
        frame_0 = cv2.imread(img_list_from_begin_label[0])
        label_0 = cv2.imread(os.path.join(label_dir, begin_label), 0)
        instance_num = max(label_0.max(),instance_num)
        label_0 = cv2.resize(label_0, (1280, 720), interpolation=cv2.INTER_NEAREST)
        frame_0 = cv2.resize(frame_0, (1280, 720))
        for i in range(1, instance_num + 1):
            if i in np.unique(label_0) and i not in search_instance:
                all_mask = {}
                x10,y10,x20,y20 = gen_bbox(label_0, i)
                max_iou = 0
                max_iou1 = 0
                max_iou2 = 0
                max_seg_iou = 0
                max_seg_iou1 = 0
                max_seg_iou2 = 0
                category =''
                category1 =''
                category2 =''
                category_seg =''
                category1_seg =''
                category2_seg =''
                max_box = '' 
                max_box1 = '' 
                max_box2 = '' 
                max_box_seg = '' 
                max_box1_seg = '' 
                max_box2_seg = '' 
                for item in det_result:
                    item = json.loads(item)
                    image_id = img_list_from_begin_label[0].replace('JPEGImages/','coco_format_imgs/JPEGImages_').replace('/00','_00')
                    image_id_num = image_id[-9:-4]
                    if item['image_id'] == image_id:
                        x1, y1, w, h = [int(iitem) for iitem in item['bbox']]
                        mask_bbox = [x1, y1, w+x1, h+y1]
                        iou = IoU(mask_bbox, [x10,y10,x20,y20])
                        if iou >= max_iou:
                            max_iou = iou
                            category = item['category_id']
                            max_box = mask_bbox
                        seg = maskUtils.decode(item['segmentation'])
                        seg_gt = label_0.copy()
                        seg_gt[seg_gt!=i]=0
                        seg_gt[seg_gt==i]=1
                        seg_iou = seg_IoU(seg_gt,np.array(seg))
                        if seg_iou > max_seg_iou:
                            max_seg_iou = seg_iou
                            category_seg = item['category_id']
                            max_box_seg = mask_bbox
                    if item['image_id'] == image_id.replace(image_id_num,'%05d' % (int(image_id_num)+1)):
                        x1, y1, w, h = [int(iitem) for iitem in item['bbox']]
                        mask_bbox = [x1, y1, w+x1, h+y1]
                        warp_prob1 = cv2.imread(wrapprob1,0)
                        warp_prob1 = cv2.resize(warp_prob1, (1280, 720), interpolation=cv2.INTER_NEAREST)
                        if i in np.unique(warp_prob1):
                            iou = IoU(mask_bbox, gen_bbox(warp_prob1, i))
                            if iou >= max_iou1:
                                max_iou1 = iou
                                category1 = item['category_id']
                                max_box1 = mask_bbox
                            seg = maskUtils.decode(item['segmentation'])
                            seg_gt = warp_prob1.copy()
                            seg_gt[seg_gt!=i]=0
                            seg_gt[seg_gt==i]=1
                            seg_iou = seg_IoU(seg_gt,np.array(seg))
                            if seg_iou > max_seg_iou1:
                                max_seg_iou1 = seg_iou
                                category1_seg = item['category_id']
                                max_box1_seg = mask_bbox
                    if item['image_id'] == image_id.replace(image_id_num,'%05d' % (int(image_id_num)+2)):
                        x1, y1, w, h = [int(iitem) for iitem in item['bbox']]
                        mask_bbox = [x1, y1, w+x1, h+y1]
                        warp_prob2 = cv2.imread(wrapprob2,0)
                        warp_prob2 = cv2.resize(warp_prob2, (1280, 720), interpolation=cv2.INTER_NEAREST)
                        if i in np.unique(warp_prob2):
                            iou = IoU(mask_bbox, gen_bbox(warp_prob2, i))
                            if iou >= max_iou2:
                                max_iou2 = iou
                                category2 = item['category_id']
                                max_box2 = mask_bbox
                            seg = maskUtils.decode(item['segmentation'])
                            seg_gt = warp_prob2.copy()
                            seg_gt[seg_gt!=i]=0
                            seg_gt[seg_gt==i]=1
                            seg_iou = seg_IoU(seg_gt,np.array(seg))
                            if seg_iou > max_seg_iou2:
                                max_seg_iou2 = seg_iou
                                category2_seg = item['category_id']
                                max_box2_seg = mask_bbox

                max_arr = [0, max_seg_iou, 0, max_seg_iou1, 0, max_seg_iou2]
                max_category = [category, category_seg, category1, category1_seg, category2, category2_seg]
                max_bbox = [max_box, max_box_seg, max_box1, max_box1_seg, max_box2, max_box2_seg]
                print(max_arr)
                print(max_category)
                #print(max_bbox)
                max_iou = max(max_arr)
                category = max_category[max_arr.index(max_iou)]
                max_box = max_bbox[max_arr.index(max_iou)]
                max_id = max_arr.index(max_iou)//2
                print(max_iou, category, max_box, max_id)

                if max_iou < float(sys.argv[3]):
                    search_instance.append(i)
                    continue

                if category == 1:
                    print('this is person')
                    search_instance.append(i)
                    continue
                frame_query = cv2.imread(img_list_from_begin_label[max_id])
                frame_query = cv2.resize(frame_query, (1280, 720))
                [x1, y1, x2, y2] = max_box
                cv2.imwrite(sys.argv[3] + sys.argv[4] + sys.argv[1] + '/query_roi.png',frame_query[y1:y2,x1:x2,:])
                query_img = Image.open(sys.argv[3] + sys.argv[4] + sys.argv[1] + '/query_roi.png').convert('RGB')
                query_img = Variable(test_transformer(query_img).unsqueeze(0)).cuda()                

                for item in det_result:
                    item = json.loads(item)
                    x1, y1, w, h = [int(iitem) for iitem in item['bbox']]
                    if w>3 and h>3 and item["category_id"] == category:
                        gallery_img = cv2.imread(item['image_id'])
                        gallery_img = cv2.resize(gallery_img, (1280, 720))
                        cv2.imwrite(sys.argv[3] + sys.argv[4] +sys.argv[1] + '/gallery_roi.png',gallery_img[y1:y1+h,x1:x1+w,:])
                        gallery_img = Image.open(sys.argv[3] + sys.argv[4] + sys.argv[1] + '/gallery_roi.png').convert('RGB')
                        gallery_img = Variable(test_transformer(gallery_img).unsqueeze(0)).cuda()
                        result = model(torch.cat([query_img,gallery_img], dim=0))
                        result =  F.normalize(result, dim=1)
                        result = result[0]-result[1]
                        score = float(torch.sqrt(torch.pow(result, 2).sum()).data.cpu())
                        if score < float(sys.argv[4]):
                            if item['image_id'] not in all_mask:
                                all_mask[item['image_id']] = [score, x1, y1, h, w, item['segmentation']]
                            elif score > all_mask[item['image_id']][0]:
                                all_mask[item['image_id']] = [score, x1, y1, h, w, item['segmentation']]
                sort_all_mask = sorted(all_mask.items(), key=lambda e:e[1][0], reverse=False)
                sort_all_mask = [(name, sim[1:]) for name, sim in sort_all_mask]
                json.dump(sort_all_mask, open(sys.argv[3] + sys.argv[4] + 'object_json/' + video_dir + '_%d.json' % i, "w"))
                search_instance.append(i)
