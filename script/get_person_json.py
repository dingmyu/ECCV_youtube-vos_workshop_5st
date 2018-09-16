import sys
sys.path.insert(0,'/mnt/lustre/sunpeng/Research/video-seg-workshop/reid/person_search/tools/')
import matplotlib.pyplot as plt
import _init_paths
import argparse
import time
import os
import sys
import os.path as osp
from glob import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import caffe
from mpi4py import MPI

from fast_rcnn.test_probe import demo_exfeat
from fast_rcnn.test_gallery import demo_detect
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list

from os.path import *
import cv2
from PIL import Image
import json
import flow as flo
import pycocotools.mask as maskUtils

def IoU(bbox1, bbox2):
    s1 = max(0, (min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + 1)) * max(0, (min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + 1))
    s2 = (max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1] + 1)) * (max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0] + 1))
    return float(s1) / float(s2)

cfg_from_file('/mnt/lustre/sunpeng/Research/video-seg-workshop/reid/person_search/experiments/cfgs/resnet50.yml')

caffe.mpi_init()
caffe.set_mode_gpu()
caffe.set_device(0)
net_query = caffe.Net('/mnt/lustre/sunpeng/Research/video-seg-workshop/reid/person_search/models/psdb/resnet50/eval_probe.prototxt', '/mnt/lustre/sunpeng/Research/video-seg-workshop/reid/person_search/output/psdb_train/resnet50/resnet50_iter_50000.caffemodel', caffe.TEST)
net_gallery = caffe.Net('/mnt/lustre/sunpeng/Research/video-seg-workshop/reid/person_search/models/psdb/resnet50/eval_gallery.prototxt', '/mnt/lustre/sunpeng/Research/video-seg-workshop/reid/person_search/output/psdb_train/resnet50/resnet50_iter_50000.caffemodel', caffe.TEST)

JPEG_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/final_test_set/test_all_frames/JPEGImages/"
Annotations_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/final_test_set/test_all_frames/mat_label/"
flow_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/final_test_set/test_all_frames/flows/flow_mean/"
inverse_flow_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/final_test_set/test_all_frames/flows/flow_inverse_mean/"
prob_dir = '/mnt/lustre/sunpeng/Research/video-seg-workshop/inference/test_set_inference/submit_baseline_2s/results_muti_label/result/'
json_dir = '/mnt/lustre/sunpeng/Research/pytorch-object-detection/examples/mask-rcnn/coco/ResNet101-FPN-Mask-16/results_test_dir/results/'

def gen_bbox(label, num):
    y, x = np.where(label == num)
    w = x.max()- x.min()
    h = y.max() - y.min()
    x1,x2,y1,y2 =  max(x.min()- int(0.05*w),0),min(x.max()+int(0.05*w),label.shape[1]-1), max(y.min()-int(0.05*h),0), min(y.max()+int(0.05*h),label.shape[0]-1)
    return [x1,y1,x2,y2]

def gen_all_bbox(label, instance_list, enlarge=False, ratio=1.0):
    bbox = np.zeros((len(instance_list), 4), float)
    bbox_enlarge = 0.15 if enlarge else 0.0

    for i in instance_list:
        [x, y] = np.where(label[:, :] == i + 1)
        if len(y) > 0:
            y = sorted(y)
            x = sorted(x)
            wmin = y[int((len(y) - 1) * (1 - ratio))]
            wmax = y[int((len(y) - 1) * (ratio))] + 1
            hmin = x[int((len(x) - 1) * (1 - ratio))]
            hmax = x[int((len(x) - 1) * (ratio))] + 1
        else:
            bbox[i, :] = [0, 0, 1, 1]
            continue

        bbox_h = hmax - hmin
        bbox_w = wmax - wmin

        wmin = np.clip((wmin - bbox_enlarge * bbox_w), 0, label.shape[1] - 1)
        wmax = np.clip((wmax + bbox_enlarge * bbox_w), wmin + 1, label.shape[1])
        hmin = np.clip((hmin - bbox_enlarge * bbox_h), 0, label.shape[0] - 1)
        hmax = np.clip((hmax + bbox_enlarge * bbox_h), hmin + 1, label.shape[0])

        bbox[i, :] = [int(wmin), int(hmin), int(wmax), int(hmax)]

    return bbox.astype(int)

def seg_IoU(seg1, seg2):
    inter = (seg1*seg2).sum()
    union = (seg1 + seg2 - seg1 * seg2).sum()
    return float(inter) / float(union)


def label_to_prob(label, channels):
    prob = np.zeros(label.shape + (channels * 2, ))
    for i in range(channels):
        prob[(label == i + 1), i * 2 + 1] = 1
        prob[(label != i + 1), i * 2] = 1
    return prob


def combine_prob(prob):
    temp_prob = np.zeros(prob.shape[0:2] + (prob.shape[2] // 2 + 1, ))
    temp_prob[..., 0] = 1
    for i in range(1, temp_prob.shape[2]):
        temp_prob[..., i] = prob[..., i * 2 - 1]
        temp_prob[..., 0] *= prob[..., i * 2 - 2]

    temp_prob = temp_prob / np.sum(temp_prob, axis=2)[..., np.newaxis]
    return temp_prob

def prob_to_label(prob):                                 
    label = np.argmax(prob, axis=2)    
    return label                  
                                                                                        
video_dir_all = os.listdir(prob_dir)
video_dir_all.sort() 
if not os.path.exists(sys.argv[3] + 'person_json/'):
    os.mkdir(sys.argv[3] + 'person_json/')
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
    my_h = 720
    my_w = 1280
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
    for (begin_label, end_label) in gt_labels_tuple:
        img_list = sorted( glob( join(frame_fr_dir, '*.jpg') ) )
        start_path = join(frame_fr_dir, begin_label.replace(".png",".jpg"))
        start_index = img_list.index(start_path) #+ 1
        img_list_from_begin_label = img_list[start_index:]
        img_list_excpet_0 = img_list[start_index + 1:]
        frames_num = len(img_list_from_begin_label)
        frame_0 = cv2.imread(img_list_from_begin_label[0])
        label_0 = cv2.imread(os.path.join(label_dir, begin_label), 0)
        label_0 = cv2.resize(label_0, (1280, 720), interpolation=cv2.INTER_NEAREST)
        frame_0 = cv2.resize(frame_0, (1280, 720))
        print(img_list_from_begin_label[1], head_num)
        wrapprob1 = prob_dir + video_dir + '/%05d.png' % (int(img_list_from_begin_label[1][-9:-4]) - head_num)
        wrapprob2 = prob_dir + video_dir + '/%05d.png' % (int(img_list_from_begin_label[2][-9:-4]) - head_num)
        #wrapprob1 = prob_dir + video_dir + '/%05d.png' % (int(img_list_from_begin_label[1][-9:-4]))
        #wrapprob2 = prob_dir + video_dir + '/%05d.png' % (int(img_list_from_begin_label[2][-9:-4]))
        instance_num = max(label_0.max(),instance_num)  
        print(instance_num,'~~~~')
        for i in range(1, instance_num + 1):
            if i in np.unique(label_0) and i not in search_instance:
                print (i)
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

                #print(max_iou, category, img_list_from_begin_label[0])
                #print(img_list_from_begin_label[1:])
                if category != 1:
                    print('this is not person')
                    search_instance.append(i)
                    continue
                elif max_iou < 0.4:
                    search_instance.append(i)
                    continue  

                query_roi = max_box
                query_img = img_list_from_begin_label[max_id]
                query_feat = demo_exfeat(net_query, query_img, query_roi)
                
                demo_detect(net_gallery, query_img)
                for gallery_img in img_list_from_begin_label[1:]:
                    boxes, features = demo_detect(net_gallery, gallery_img,
                                      threshold=0.7)
                    if boxes is None:
                        print (gallery_img, 'no detections')
                    else:           
                        similarities = features.dot(query_feat)
                        myList = zip(boxes, similarities)
                        box, sim = sorted(myList, key=lambda x:x[1])[-1]
                        x1, y1, x2, y2, _ = box
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        if sim > float(sys.argv[3]):
                            max_iou = 0
                            category =''
                            max_box = '' 
                            seg_mask =''
                            for item in det_result:
                                item = json.loads(item)
                                image_id = gallery_img.replace('JPEGImages/','coco_format_imgs/JPEGImages_').replace('/00','_00')
                                if item['image_id'] == image_id:
                                    xx, yy, w, h = [int(iitem) for iitem in item['bbox']]
                                    if w>5 and h > 5:
                                        mask_bbox = [xx, yy, w+xx, h+yy]
                                        iou = IoU(mask_bbox, [x1, y1, x2, y2])
                                        if iou >= max_iou:
                                            max_iou = iou
                                            category = item['category_id']
                                            max_box = mask_bbox
                                            seg_mask = item['segmentation']
                            if max_iou > float(sys.argv[3]):
                                if gallery_img not in all_mask:
                                    all_mask[gallery_img] = [sim, x1, y1, x2, y2, seg_mask]
                                    #plt.imshow(np.array(maskUtils.decode(item['segmentation'])))
                                elif sim > all_mask[gallery_img][0]:
                                    all_mask[gallery_img] = [sim, x1, y1, x2, y2, seg_mask]
                                #plt.imshow(np.array(maskUtils.decode(item['segmentation'])))
                sort_all_mask = sorted(all_mask.items(), key=lambda e:e[1][0], reverse=True)
                sort_all_mask = [(name, sim[1:]) for name, sim in sort_all_mask]
                #print(sort_all_mask)
                json.dump(sort_all_mask, open(sys.argv[3] + 'person_json/' + video_dir + '_%d.json' % i, "w"))
                search_instance.append(i)
