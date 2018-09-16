import sys
import matplotlib.pyplot as plt
import argparse
import time
import os
import sys
import os.path as osp
from glob import glob
import numpy as np
from os.path import *
import cv2
from PIL import Image
import json
import flow as flo
import json
import pycocotools.mask as maskUtils

import torch
from torchE.D import dist_init, average_gradients, DistModule
from torch.autograd import Variable
from models.pspnet2S import PSP2S
import torch.nn.parallel
import torch.backends.cudnn as cudnn

model = PSP2S(backbone='resnet', layers=101, classes=2, zoom_factor=1, syncbn=False, pretrained=False).cuda()
model = torch.nn.DataParallel(model).cuda()
cudnn.enabled = True
cudnn.benchmark = True
#weights = '/mnt/lustre/sunpeng/Research/video-seg-workshop/models/deeplabv3_models/final_2s_seg_youtube_online_two/exp/drivable/youtube_2s_2pairs_4epoch_fix_final_mask_0809/model/train_epoch_2.pth'
weights = '/mnt/lustre/sunpeng/Research/video-seg-workshop/models/deeplabv3_models/final_2s_seg_youtube_online_two/exp/drivable/youtube_2s_2pairs_0825/model/train_epoch_6.pth'
print(("=> loading checkpoint '{}'".format(weights)))
checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['state_dict'], strict=False)
print(("=> loaded checkpoint"))
model.eval()

def IoU(bbox1, bbox2):
    s1 = max(0, (min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + 1)) * max(0, (min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + 1))
    s2 = (max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1] + 1)) * (max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0] + 1))
    return float(s1) / float(s2)

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
def enlarge_bbox(box, enlarge=0.15):
    [x1, y1, x2, y2] = box
    w = x2-x1
    h = y2-y1
    x1 = np.clip(x1 - enlarge*w, 0, x1)
    x2 = np.clip(x2 + enlarge*w, x2, my_w-1)
    y1 = np.clip(y1 - enlarge*h, 0, y1)
    y2 = np.clip(y2 + enlarge*h, y2, my_h-1)
    return [int(x1), int(y1), int(x2), int(y2)]
    


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


#if not os.path.exists('Annotations'):
#    os.mkdir('Annotations')
video_dir_all = os.listdir(prob_dir)
video_dir_all.sort()
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
        gt_labels_tuple = list(zip(gt_labels, gt_labels_b))
    head_num = int(gt_labels_tuple[0][0][:5])
    #print(gt_labels_tuple,head_num)
    search_instance = []

    #print(small_num)
    instance_num = 0

    img_list = sorted( glob( join(frame_fr_dir, '*.jpg') ) )
    #print(img_list)
    begin_label = gt_labels_tuple[0][0]
    start_path = join(frame_fr_dir, begin_label.replace(".png",".jpg"))
    #print(start_path)
    start_index = img_list.index(start_path) #+ 1
    img_list_excpet_0 = img_list[start_index + 1:]
    img_list_from_begin_label = img_list[start_index:]
    frame_0 = cv2.imread(img_list_from_begin_label[0])
    frame_0 = cv2.resize(frame_0, (1280, 720))
    frames_num = len(img_list_from_begin_label)
    patch_shape = (433, 433)

    frames = [None for _ in range(frames_num)]
    flow1 = [None for _ in range(frames_num)]
    flow2 = [None for _ in range(frames_num)]
    frames[0] = frame_0 #cv2.resize(frames[0], (my_w, my_h))
    #bbox = gen_all_bbox(label_0, range(instance_num), True)

    th = 1
    for name in img_list_excpet_0:
        frames_th = cv2.imread(name)
        frames[th] = cv2.resize(frames_th, (my_w, my_h)) # add by sunpeng hard code.
        flow1[th - 1] = flo.readFlow(os.path.join(flow_dir1, '%06d.flo' % (start_index + th - 1)))
        flow2[th] = flo.readFlow(os.path.join(flow_dir2, '%06d.flo' % (start_index + th -1)))
        th += 1    

    #print(len(flow1),len(flow2))
    pic_name_list = os.listdir(prob_dir + video_dir)
    pic_name_list.sort()
    small_num = int(img_list[0][-9:-4])
    prob_num = len(pic_name_list)
    print(prob_num)
    pic_list= []    
    for index, probname in enumerate(pic_name_list):
        pic = cv2.imread(prob_dir + video_dir + '/' + probname, 0)
        pic = cv2.resize(pic, (1280, 720), interpolation=cv2.INTER_NEAREST)
        pic_list.append(pic)


    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]


    def propgate_forward(init_num, channel, gallery_img):
        gallery_img = gallery_img.replace(gallery_img[-9:-4], '%05d' % (int(gallery_img[-9:-4])+1))
        prob = pic_list[init_num]
        if init_num >= prob_num - 1:
            print('end')
            return
        if channel not in np.unique(prob):
            print('cant follow')
            return
        if channel in np.unique(pic_list[init_num + 1]):
            print('already have')
            return
        print('propgate_forward',init_num)
        tmp_prob = combine_prob(label_to_prob(prob, channel))
        #print(tmp_prob.shape)
        warp_prob = flo.get_warp_label(flow1[init_num], flow2[init_num + 1], tmp_prob)
        #print(warp_prob.shape)
        warp_label = prob_to_label(warp_prob)  
        if channel not in np.unique(warp_label):
            print('cant follow')
            return    
    #     plt.imshow(warp_label)
    #     plt.axis('off')
    #     plt.tight_layout()    
    #     plt.show()   
        x1, y1, x2, y2 = enlarge_bbox(gen_bbox(warp_label, channel))
        if x2-x1 < 3:
            return
        if y2-y1 < 3:
            return
        image_patch = cv2.imread(gallery_img).astype(np.float32)
        image_patch = cv2.resize(image_patch, (1280, 720))[y1:y2,x1:x2,:]
        warp_label_patch = warp_label[y1:y2,x1:x2]
        warp_label_patch[warp_label_patch!=channel] = 0
        warp_label_patch[warp_label_patch==channel] = 1
        warp_label_patch_tmp = np.zeros((y2-y1,x2-x1,1))
        warp_label_patch_tmp[:,:,0] = warp_label_patch

        flow_patch = flow2[init_num + 1][y1:y2,x1:x2,:]
        image_patch = cv2.resize(image_patch,patch_shape)
        warp_label_patch = cv2.resize(warp_label_patch_tmp,patch_shape)
        flow_patch = cv2.resize(flow_patch,patch_shape)     
        image_patch = torch.from_numpy(image_patch.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
        for t, m, s in zip(image_patch[0], mean, std):
            t.sub_(m).div_(s)

        warp_label_patch = torch.from_numpy(warp_label_patch).contiguous().float().unsqueeze(0).unsqueeze(0).cuda()
        flow_patch = torch.from_numpy(flow_patch.transpose(2, 0, 1)).contiguous().float().unsqueeze(0).cuda()

        print(image_patch.size(), warp_label_patch.size(), flow_patch.size())
        new_prob = model(image_patch, warp_label_patch, flow_patch)
        print(new_prob.size())
        new_prob = torch.nn.functional.softmax(new_prob, dim=1).squeeze(0)[1].data.cpu().numpy()
        new_prob[new_prob>0.5] = 1
        new_prob[new_prob<=0.5] = 0
        tmp_new_prob = np.zeros((433,433,1))
        tmp_new_prob[:,:,0] = new_prob
        new_prob = cv2.resize(tmp_new_prob,(x2-x1,y2-y1))
        prob = pic_list[init_num + 1] 
        prob[y1:y2,x1:x2][(prob[y1:y2,x1:x2] == 0) & (new_prob==1)] = channel
        pic_list[init_num + 1] = prob
        propgate_forward(init_num + 1, channel, gallery_img)


    def propgate_backward(init_num, channel, gallery_img, minid):
        gallery_img = gallery_img.replace(gallery_img[-9:-4], '%05d' % (int(gallery_img[-9:-4])-1))
        prob = pic_list[init_num]
        if init_num <= minid:
            print('end')
            return
        if channel not in np.unique(prob):
            print('cant follow')
            return
        if channel in np.unique(pic_list[init_num - 1]):
            print('already have')
            return
        print('propgate_backward',init_num)  
        tmp_prob = combine_prob(label_to_prob(prob, channel))
        #print(tmp_prob.shape)
        warp_prob = flo.get_warp_label(flow2[init_num],flow1[init_num - 1], tmp_prob)
        #print(warp_prob.shape)
        warp_label = prob_to_label(warp_prob)  
        if channel not in np.unique(warp_label):
            print('cant follow')
            return    

        x1, y1, x2, y2 = enlarge_bbox(gen_bbox(warp_label, channel))
        if x2-x1 < 3:
            return
        if y2-y1 < 3:
            return
        image_patch = cv2.imread(gallery_img).astype(np.float32)
        image_patch = cv2.resize(image_patch, (1280, 720))[y1:y2,x1:x2,:]
        warp_label_patch = warp_label[y1:y2,x1:x2]
        warp_label_patch[warp_label_patch!=channel] = 0
        warp_label_patch[warp_label_patch==channel] = 1
        warp_label_patch_tmp = np.zeros((y2-y1,x2-x1,1))
        warp_label_patch_tmp[:,:,0] = warp_label_patch

        flow_patch = flow1[init_num-1][y1:y2,x1:x2,:]
        image_patch = cv2.resize(image_patch,patch_shape)
        warp_label_patch = cv2.resize(warp_label_patch_tmp,patch_shape)
        flow_patch = cv2.resize(flow_patch,patch_shape)

        image_patch = torch.from_numpy(image_patch.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
        for t, m, s in zip(image_patch[0], mean, std):
            t.sub_(m).div_(s)

        warp_label_patch = torch.from_numpy(warp_label_patch).contiguous().float().unsqueeze(0).unsqueeze(0).cuda()
        flow_patch = torch.from_numpy(flow_patch.transpose(2, 0, 1)).contiguous().float().unsqueeze(0).cuda()

        print(image_patch.size(), warp_label_patch.size(), flow_patch.size())
        new_prob = model(image_patch, warp_label_patch, flow_patch)
        print(new_prob.size())
        new_prob = torch.nn.functional.softmax(new_prob, dim=1).squeeze(0)[1].data.cpu().numpy()
        new_prob[new_prob>0.5] = 1
        new_prob[new_prob<=0.5] = 0
        tmp_new_prob = np.zeros((433,433,1))
        tmp_new_prob[:,:,0] = new_prob
        new_prob = cv2.resize(tmp_new_prob,(x2-x1,y2-y1))
        prob = pic_list[init_num - 1] 
        prob[y1:y2,x1:x2][(prob[y1:y2,x1:x2] == 0) & (new_prob==1)] = channel
        pic_list[init_num - 1] = prob

        propgate_backward(init_num - 1, channel, gallery_img, minid)


    for (begin_label, end_label) in gt_labels_tuple:
        img_list = sorted( glob( join(frame_fr_dir, '*.jpg') ) )
        #print(img_list)
        start_path = join(frame_fr_dir, begin_label.replace(".png",".jpg"))
        #print(start_path)
        start_index = img_list.index(start_path) #+ 1
        #print(start_index)

        img_list_from_begin_label = img_list[start_index:]
        img_list_excpet_0 = img_list[start_index + 1:]
        frame_0 = cv2.imread(img_list_from_begin_label[0])
        label_0 = cv2.imread(os.path.join(label_dir, begin_label), 0)
        label_0 = cv2.resize(label_0, (1280, 720), interpolation=cv2.INTER_NEAREST)
        frame_0 = cv2.resize(frame_0, (1280, 720))
        instance_num = max(label_0.max(), instance_num)
        print(instance_num, '~~~~~~~~~~~~~~~~')
        for i in range(1, instance_num + 1):
            if i in np.unique(label_0) and i not in search_instance:
                print (i)
                all_mask = {}
                query_roi = gen_bbox(label_0, i)
                query_img = img_list_from_begin_label[0]

                if not os.path.exists(sys.argv[3] + 'object_json/' + video_dir + '_%d.json' % i):
                    print('this is not person')
                    search_instance.append(i)
                    continue
                sort_all_mask = json.loads(open(sys.argv[3] +'object_json/' + video_dir + '_%d.json' % i).read())
                for gallery_img, (x1, y1, x2, y2, mask) in sort_all_mask:
                    x1, y1, x2, y2 = enlarge_bbox([x1, y1, x2, y2])
                    mask = np.array(maskUtils.decode(mask))

                    #prob_name = prob_dir + video_dir + '/%05d.png' % (int(gallery_img[-9:-4]) - head_num)
                    if int(gallery_img[-9:-4]) - small_num < start_index:
                        continue
                    prob = pic_list[int(gallery_img[-9:-4]) - head_num]
                    if i in np.unique(prob):
                        continue
                    print (gallery_img, x1, y1, x2, y2)               
                    prob[(prob==0) & (mask==1)] = i 
                    #prob = prob + mask * i
                    pic_list[int(gallery_img[-9:-4]) - head_num] = prob

                    propgate_forward(int(gallery_img[-9:-4]) - head_num, i, gallery_img)
                    propgate_backward(int(gallery_img[-9:-4]) - head_num, i, gallery_img, start_index - head_num + small_num)

                search_instance.append(i)        
    os.mkdir(sys.argv[3] +'Annotations/' + video_dir)
    for index, probname in enumerate(pic_name_list):
        pic = pic_list[index]
        cv2.imwrite(sys.argv[3] +'Annotations/' + video_dir + '/' + probname, pic)
