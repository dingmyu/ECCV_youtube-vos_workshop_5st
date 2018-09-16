import os
import torch
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from core.models.pspnet2S import PSP2S#from pspnet import PSPNet
from core.utils import flow as flo
from core.utils.disp import labelcolormap
from core.utils.bbox import gen_bbox, label_to_prob, combine_prob, prob_to_label, IoU
from core.utils.pickle_io import pickle_dump, pickle_load
import cv2
import torch
from torch.autograd import Variable
from os.path import *
from glob import glob
import os
patch_shape = (433, 433)

use_flip = True
bbox_occ_th = 0.3
reid_th = 0.5
mask_th = 0.5

my_h = 720
my_w = 1280

count = 0

def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda(0))
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())

def predict(st, en, step, instance_list):
    # st = 1, step =1, instance_list: obj number id.
    global pred_prob
    global bbox_cnt
    global count

    for th in range(st, en, step):
        if (step == 1):
            temp_prob_ = combine_prob(pred_prob[th - 1])
            # print (temp_prob_.shape) # (720, 1280, 3)
            # temp_prob_[temp_prob_ > mask_th] = 1
            warp_prob = flo.get_warp_label(flow1[th - 1], flow2[th], temp_prob_)
        elif (step == -1):
            temp_prob_ = combine_prob(pred_prob[th + 1])
            # temp_prob_[temp_prob_ > mask_th] = 1
            warp_prob = flo.get_warp_label(flow2[th + 1], flow1[th], temp_prob_)
        bbox = gen_bbox(prob_to_label(warp_prob), range(instance_num), True)


        new_instance_list = []
        abort = True

        temp = prob_to_label(combine_prob(pred_prob[th - step]))

        for i in range(instance_num):
            if (th == st) or (np.count_nonzero(orig_mask[i]) <= np.count_nonzero(temp == (i + 1)) * 10):
                if np.abs(bbox_cnt[th][i] - th) <= np.abs((st - step) - th):
                    continue
                if i in instance_list:
                    abort = False
                    bbox_cnt[th][i] = (st - step)
                    new_instance_list.append(i)
                else:
                    for j in instance_list:
                        if IoU(bbox[i, :], bbox[j, :]) > 1e-6:
                            new_instance_list.append(i)
                            break

        if abort:
            break

        new_instance_list = sorted(new_instance_list)
        temp_image = frames[th].astype(float)

        f_prob = [np.zeros([bbox[idx, 3] - bbox[idx, 1], bbox[idx, 2] - bbox[idx, 0], 2]) for idx in new_instance_list]
        image_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 3), float)
        flow_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 2), float)
        warp_label_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 1), float)

        for i in range(len(new_instance_list)):
            idx = new_instance_list[i]
            warp_label_patch_temp = cv2.resize(warp_prob[bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx + 1], patch_shape).astype(float)
            # print (warp_label_patch_temp.shape) # (433, 433)
            warp_label_patch_temp[warp_label_patch_temp >= mask_th] = 1
            warp_label_patch_temp[warp_label_patch_temp < mask_th] = 0
            warp_label_patch[i, ..., 0] = warp_label_patch_temp
            image_patch[i, ...] = cv2.resize(temp_image[int(0.5 + bbox[idx, 1] * fr_h_r):int(0.5 + bbox[idx, 3] * fr_h_r),
                                                        int(0.5 + bbox[idx, 0] * fr_w_r):int(0.5 + bbox[idx, 2] * fr_w_r), :], patch_shape).astype(float)
            if (step == 1):
                flow_patch[i, ...] = cv2.resize(flow2[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], :], patch_shape).astype(float)
            else:
                flow_patch[i, ...] = cv2.resize(flow1[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], :], patch_shape).astype(float)


        image_patch = torch.from_numpy(image_patch.transpose(0, 3, 1, 2))
        warp_label_patch = torch.from_numpy(warp_label_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()
        flow_patch = torch.from_numpy(flow_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()

        print(image_patch.size(),'image_patch')
        print(warp_label_patch.size(),'warp_label_patch')
        print(flow_patch.size(),'flow_patch')
        
        #notice here!!!!!!! if annotate it, you will be mad.

        print((image_patch[0]).cpu().numpy().transpose(1, 2, 0).shape)
        cv2.imwrite('output/%3d_prob.png' % count, (image_patch[0]).cpu().numpy().transpose(1, 2, 0))
        count += 1   
        cv2.imwrite('output/%3d_prob.png' % count, (warp_label_patch[0][0]*255).cpu().numpy())
        count += 1 

        
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        for i in range(len(new_instance_list)):
            for t, m, s in zip(image_patch[i], mean, std):
                t.sub_(m).div_(s)
        # notice here!!!!!!! if annotate it, you will be mad.

        image_patch = image_patch.contiguous().float().cuda()

        with torch.no_grad():
            prob = model(image_patch, warp_label_patch, flow_patch) #model = torch.nn.DataParallel(model).cuda()
            prob = torch.nn.functional.softmax(prob, dim=1)
            
            cv2.imwrite('output/%3d_prob.png' % count, (torch.argmax(prob[0], dim=0, keepdim=False)).cpu().numpy()*100)
            count += 1       
            
         # warp_label_patch debug1
#         probxxx = prob.data.cpu().numpy().transpose(0, 2, 3, 1)
#         print (probxxx[0].shape)
#         out = np.argmax(probxxx[0], axis=2)
#         print (out.shape)
#         cv2.imwrite("debugs/"+str(th)+"_out.png", (out * 255))
        

        if use_flip:
            image_patch = flip(image_patch, 3)
            warp_label_patch = flip(warp_label_patch, 3)
            flow_patch = flip(flow_patch, 3)
            flow_patch[:, 0, ...] = -flow_patch[:, 0, ...]
            with torch.no_grad():
                prob_f = model(image_patch, warp_label_patch, flow_patch)
                #cv2.imwrite('output/%d_probf.png' % count, (prob_f[0][0]*255).cpu().numpy())
                #count += 1
            prob_f = torch.nn.functional.softmax(prob_f, dim=1)
            prob_f = flip(prob_f, 3)
            prob = (prob + prob_f) / 2.0

        prob = prob.data.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(len(new_instance_list)):
            idx = new_instance_list[i]
            f_prob[i] += cv2.resize(prob[i, ...], (bbox[idx, 2] - bbox[idx, 0], bbox[idx, 3] - bbox[idx, 1]))

        for i in range(len(new_instance_list)):
            idx = new_instance_list[i]
            pred_prob[th][..., idx * 2] = 1
            pred_prob[th][..., idx * 2 + 1] = 0
            pred_prob[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx * 2] = f_prob[i][..., 0]
            pred_prob[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx * 2 + 1] = f_prob[i][..., 1]

def update_appear():
    global pred_prob
    global appear
    global location

    for th in range(appear.shape[0]):
        bbox = gen_bbox(prob_to_label(combine_prob(pred_prob[th])), range(instance_num))
        for i in range(appear.shape[1]):
            appear[th, i] = ((bbox[i, 2] - bbox[i, 0]) * (bbox[i, 3] - bbox[i, 1]) > 1)
            if appear[th, i] > 0:
                location[th, i, 0] = float(bbox[i, 2] + bbox[i, 0]) / 2
                location[th, i, 1] = float(bbox[i, 3] + bbox[i, 1]) / 2
            else:
                location[th, i, :] = location[th - 1, i, :]


def parse_args():
    parser = argparse.ArgumentParser(description='Train Segmentation')
    # ========================= Model Configs ==========================
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--weights', type=str, default='/mnt/lustre/sunpeng/Research/video-seg-workshop/models/deeplabv3_models/final_2s_seg_youtube_online_two/exp/drivable/youtube_2s_2pairs_4nodes/model/train_epoch_10.pth')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cache', type=str, default='')
    parser.add_argument('--json', type=str, default='')
    args = parser.parse_args()
    return args

def main():
    def save_frame_mine(prob, total_index ,dir_name='', video_name='' , vis=True):
        result = prob_to_label(combine_prob(prob))
        result_show = np.dstack((colors[result, 0], colors[result, 1], colors[result, 2])).astype(np.uint8)
        if args.output != '' and dir_name != '':
            out_file = os.path.join(str(dataset_dir), str(dir_name), str(video_name), '%05d.png' % total_index)
            if not os.path.exists(os.path.split(out_file)[0]):
                os.makedirs(os.path.split(out_file)[0])
            if vis:
                cv2.imwrite(out_file, result_show)
            else:
                cv2.imwrite(out_file, result)
        return

    colors = labelcolormap(256)

    global pred_prob, frames, flow1, flow2, orig_mask, \
        model, instance_num, fr_h_r, fr_w_r, appear, bbox_cnt, \
        location, patch_shapes

    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model = PSP2S(backbone='resnet', layers=101, classes=2, zoom_factor=1, syncbn=True, group_size=8, group=1).cuda()

    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(args.weights):
        print(("=> loading checkpoint '{}'".format(args.weights)))
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'], True)
        print(("=> loaded checkpoint"))
    else:
        raise (("=> no checkpoint found at '{}'".format(args.weights)))
    model = model.cuda()
    model.eval()
    cudnn.benchmark = True
    # Setup dataset
    dataset_dir = os.path.join('results_muti_label')
    JPEG_dir = "/mnt/lustre/share/sunpeng/video-seg-workshop/valid_all_frames/JPEGImages/"
    Annotations_dir = "/mnt/lustre/share/chengguangliang/for_sunpeng/valid/"
    flow_dir = "/mnt/lustre/share/sunpeng/video-seg-workshop/valid_flows/flow_mean/"
    inverse_flow_dir = "/mnt/lustre/share/sunpeng/video-seg-workshop/valid_flows/flow_inverse_mean/"
    # train set
#     JPEG_dir = "/mnt/lustre/share/sunpeng/video-seg-workshop/train_all_frames/JPEGImages/"
#     Annotations_dir = "/mnt/lustre/sunpeng/test"
#     flow_dir = "/mnt/lustre/share/sunpeng/video-seg-workshop/train_flows/flow_mean"
#     inverse_flow_dir = "/mnt/lustre/share/sunpeng/video-seg-workshop/train_flows/flow_inverse_mean/"

    # cache_dir = "/mnt/lustre/sunpeng/Research/video-seg-workshop/valid/cache/"
    # val_example = "/mnt/lustre/sunpeng/Research/video-seg-workshop/submits/val_submit/"
#     video_list = []
#     for line in os.listdir(JPEG_dir):
#         video_list.append(line.strip())
#     print ("length in val set...")
#     print (len(video_list))
    video_cnt = 0
    for video_dir in range(0,1): #open("jsons/"+args.json).readlines():
        if video_cnt % 1 == 0:
            print (video_cnt)
        video_dir = '00f88c4f0a'
#         video_dir = 'ad4108ee8e'
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

        total_index = 0
        first_come = True
        if len(gt_labels) == 1:
            gt_labels_tuple = [(gt_labels[0], -100)]
        else:
            gt_labels_b = gt_labels[1:]
            gt_labels_b.append(-100)
            gt_labels_tuple = zip(gt_labels, gt_labels_b)
        video_cnt += 1
        for (begin_label, end_label) in gt_labels_tuple:
            img_list = sorted( glob( join(frame_fr_dir, '*.jpg') ) )
            start_path = join(frame_fr_dir, begin_label.replace(".png",".jpg"))
            start_index = img_list.index(start_path) #+ 1
            if end_label == -100:
                img_list_from_begin_label = img_list[start_index:]
                img_list_excpet_0 = img_list[start_index + 1:]
            else:
                end_path = join(frame_fr_dir, end_label.replace(".png",".jpg"))
                end_index = img_list.index(end_path)
                img_list_from_begin_label = img_list[start_index:end_index]
                img_list_excpet_0 = img_list[start_index + 1:end_index]

            frame_0 = cv2.imread(img_list_from_begin_label[0])
            label_0 = cv2.imread(os.path.join(label_dir, begin_label), 0)

            frames_num = len(img_list_from_begin_label) # frame number begins the one which has labels.
            label_0 = label_0.astype(np.uint8)
            label_0 = cv2.resize(label_0, (my_w, my_h), interpolation=cv2.INTER_NEAREST)
            if len(gt_labels) != 1:
                if first_come == True:
                    first_come = False
                    pass
                else:
                    before_prediction = list(reversed(sorted(os.listdir("./results_muti_label/result/"+video_dir))))[0]
                    before_prediction = join("./results_muti_label/result/", video_dir, before_prediction)
                    label_1 = cv2.imread(before_prediction, 0)
                    label_1 = cv2.resize(label_1, (my_w, my_h), interpolation=cv2.INTER_NEAREST)
                    before_instance_number = label_1.max()
                    id = len(np.unique(label_0)) - 1
                    for i in np.unique(label_0):
                        if i != 0:
                            label_1[label_0 == i] = i
                    label_0 = label_1
                    print ("load before predict: %s, before has %s objects, load %s objects, total %s objects now" %
                                      (before_prediction,before_instance_number,id,label_0.max()))
            label_0 = label_0.astype(np.uint8)
            frame_0 = cv2.resize(frame_0, (my_w, my_h))
            # label_0 = cv2.resize(label_0, (my_w, my_h), interpolation=cv2.INTER_NEAREST)
            instance_num = label_0.max()
            print ("prepare create array...")
            frames = [None for _ in range(frames_num)]
            pred_prob = [None for _ in range(frames_num)]
            flow1 = [None for _ in range(frames_num)]
            flow2 = [None for _ in range(frames_num)]
            orig_mask = [None for _ in range(instance_num)]
            person_reid = [[None for _ in range(instance_num)] for _ in range(frames_num)]
            frames[0] = frame_0 #cv2.resize(frames[0], (my_w, my_h))
            fr_h_r = 1 #my_h #float(frames[0].shape[0]) / float(frame_0.shape[0])
            fr_w_r = 1 #my_w #float(frames[0].shape[1]) / float(frame_0.shape[1])
            pred_prob[0] = label_to_prob(label_0, instance_num)
#             save_frame_mine(pred_prob[0], total_index, 'result' ,video_dir, True)
            bbox = gen_bbox(label_0, range(instance_num), True)
            for i in range(instance_num):
                orig_mask[i] = pred_prob[0][bbox[i, 1]:bbox[i, 3], bbox[i, 0]:bbox[i, 2], i * 2 + 1]
            th = 1
            print ("all imgs frame number: %s, start gt frame:%s, folder_name is: %s ,current process: %s ==== 24 " % (frames_num, start_index, video_dir , video_cnt))
            for name in img_list_excpet_0:
                frames_th = cv2.imread(name)
                frames[th] = cv2.resize(frames_th, (my_w, my_h)) # add by sunpeng hard code.
                pred_prob[th] = label_to_prob(np.zeros_like(label_0, np.uint8), instance_num)
                flow1[th - 1] = flo.readFlow(os.path.join(flow_dir1, '%06d.flo' % (start_index + th - 1)))
                flow2[th] = flo.readFlow(os.path.join(flow_dir2, '%06d.flo' % (start_index + th -1)))
                #print(name, start_index)
#                 print ("load flow1 and flow2, %s %s" % ('%06d.flo' % (start_index + th - 1), '%06d.flo' % (start_index + th -1)))
                th += 1
            bbox_cnt = -1000 * np.ones((frames_num, instance_num))
            bbox_cnt[0, :] = 0

            predict(1, frames_num, 1, range(instance_num))

            appear = np.zeros((frames_num, instance_num)).astype(int)
            location = np.zeros((frames_num, instance_num, 2)).astype(int)
            update_appear()

            for th in range(frames_num):
                save_frame_mine(pred_prob[th], total_index, 'draft' ,video_dir , True)
                save_frame_mine(pred_prob[th], total_index, 'result', video_dir, False)
                total_index += 1
        # break

if __name__ == '__main__':
    main()
