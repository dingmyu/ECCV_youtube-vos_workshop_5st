import os
import torch
from torch.utils.data import Dataset
import cv2
import random
from tps import from_control_points
import numpy as np
cv2.ocl.setUseOpenCL(False)
import pdb

# common class
class PatchDataSet(Dataset):
    ignore_label = 255
    num_class = 36

    def __init__(self, dataset_path, data_list, norm=None, is_train=True, sub_batch=1):
        random.seed()
        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            for line in f:
                self.img_list.append(line[:-1].split(' '))

        self.dataset_path = dataset_path
        self.norm = norm
        
        self.patch_size = 54 * 8 + 1 # 432 16x+1
        self.is_train = is_train
        self.sub_batch = sub_batch

    def __len__(self):
        return len(self.img_list)

    def gen_bbox_mask(self, image, seg_label, ins_label, idx):
        aug = 0.2
        enlage = random.uniform(0.1, 0.3)

        [y, x] = np.where(ins_label == idx)

        wmax = max(x) + 1
        wmin = min(x)
        hmax = max(y) + 1
        hmin = min(y)

        bbox_h = hmax - hmin
        bbox_w = wmax - wmin

        wmin += random.randint(-int(aug * bbox_w), int(aug * bbox_w))
        wmax += random.randint(-int(aug * bbox_w), int(aug * bbox_w))
        hmin += random.randint(-int(aug * bbox_h), int(aug * bbox_h))
        hmax += random.randint(-int(aug * bbox_h), int(aug * bbox_h))

        bbox_h = hmax - hmin
        bbox_w = wmax - wmin
        assert (bbox_h >= 0)
        assert (bbox_w >= 0)

        wmin = np.clip((wmin - enlage * bbox_w), 0, ins_label.shape[1] - 1)
        wmax = np.clip((wmax + enlage * bbox_w), wmin + 1, ins_label.shape[1])
        hmin = np.clip((hmin - enlage * bbox_h), 0, ins_label.shape[0] - 1)
        hmax = np.clip((hmax + enlage * bbox_h), hmin + 1, ins_label.shape[0])

        bbox = [int(wmin), int(hmin), int(wmax), int(hmax)]
        img_patch = cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        seg_mask = cv2.resize(seg_label[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        ins_mask = cv2.resize(ins_label[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        ins_mask[(ins_mask != idx) & (ins_mask != PatchDataSet.ignore_label)] = 0
        ins_mask[(ins_mask == idx)] = 1
        # ins_mask = (ins_mask == idx).astype(np.uint8)

        return img_patch, seg_mask, ins_mask

    def aug_mask(self, ins_mask):
        if np.count_nonzero(ins_mask == 1) == 0:
            ins_mask[0][0] = 1
        [y, x] = np.where(ins_mask == 1)
        wmax = max(x)
        wmin = min(x)
        hmax = max(y)
        hmin = min(y)
        _, contour, _ = cv2.findContours(ins_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contour != []:
            contour = np.squeeze(np.vstack(contour))
        match_point = 5
        if (contour.shape[0] > match_point * 10):
            control_points = []
            for j in range(match_point):
                idx = random.randint(int(contour.shape[0] * (2 * j) / (2 * match_point)), int(contour.shape[0] * (2 * j + 1) / (2 * match_point)))
                sx, sy = contour[idx, :]
                tx = sx + random.uniform(-0.1 * (wmax - wmin + 1), 0.1 * (wmax - wmin + 1))
                ty = sy + random.uniform(-0.1 * (hmax - hmin + 1), 0.1 * (hmax - hmin + 1))
                control_points.append((sx, sy, tx, ty))

            try:
                t = from_control_points(control_points, backwards=True)
                ins_mask2 = ins_mask.copy()
                [x, y] = np.meshgrid(range(ins_mask.shape[1]), range(ins_mask.shape[0]))
                xy = np.vstack((x.flatten(), y.flatten()))
                # newxy = xy.copy()
                # for i in range(xy.shape[1]):
                newxy = np.round(np.array([t.transform(*xy[:, i]) for i in range(xy.shape[1])])).astype(int)
                newxy[:, 0] = np.clip(newxy[:, 0], 0, ins_mask.shape[1] - 1)
                newxy[:, 1] = np.clip(newxy[:, 1], 0, ins_mask.shape[0] - 1)
                ins_mask[xy[1, :], xy[0, :]] = ins_mask2[newxy[:, 1], newxy[:, 0]]
            except Exception:
                print('warp error!')

            patch_num = random.randint(1, 5)
            for i in range(patch_num):
                tot = np.count_nonzero(ins_mask)
                x = random.randint(0, ins_mask.shape[1] - 1)
                y = random.randint(0, ins_mask.shape[0] - 1)
                tot = int(tot * random.uniform(0.02, 0.05))
                key = (1 if ins_mask[y, x] == 0 else 0)
                for j in range(tot):
                    cv2.circle(ins_mask, (x, y), (2 if ins_mask[y, x] == 0 else 8), key, thickness=-1)
                    x = x + random.randint(-1, 1)
                    y = y + random.randint(-1, 1)
                    x = np.clip(x, 0, ins_mask.shape[1] - 1)
                    y = np.clip(y, 0, ins_mask.shape[0] - 1)

        dilation_size = random.randint(0, 2) * 2 + 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size * 2 + 1, dilation_size * 2 + 1), (dilation_size, dilation_size))
        ins_mask = cv2.dilate(ins_mask, element)
        return ins_mask

    def get_single_item(self, idx, b_idx):
        image = cv2.imread(os.path.join(self.dataset_path, self.img_list[idx][0])).astype(np.float32)
        seg_label = cv2.imread(os.path.join(self.dataset_path, self.img_list[idx][1]), cv2.IMREAD_UNCHANGED)
        ins_label = cv2.imread(os.path.join(self.dataset_path, self.img_list[idx][2]), cv2.IMREAD_UNCHANGED)
            
        # flip
        if (random.random() > 0.5 and self.is_train):
            image = np.fliplr(image)
            seg_label = np.fliplr(seg_label)
            ins_label = np.fliplr(ins_label)

        # rotate
        if (random.random() > 0.5 and self.is_train):
            angl = random.uniform(-10, 10)
            h, w = image.shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, angl, 1.0)
            image = cv2.warpAffine(image, map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            seg_label = cv2.warpAffine(seg_label, map_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=PatchDataSet.ignore_label)
            ins_label = cv2.warpAffine(ins_label, map_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=PatchDataSet.ignore_label)

        idxs = np.unique(ins_label)
        idxs = list(idxs[(idxs != 0) & (idxs != PatchDataSet.ignore_label)])

        if len(idxs) == 0:
            print("No Instance!!!!!!!!!!!!!!!!!!")
            return self.get_single_item(random.randint(0, self.__len__() - 1), b_idx)
        else:
            idx = random.sample(idxs, 1)

        image_patch, seg_mask, ins_mask = self.gen_bbox_mask(image, seg_label, ins_label, idx)
        if (random.random() > 0.4):
            aug_mask = self.aug_mask((ins_mask == 1).astype(np.uint8))
        else:
            aug_mask = np.zeros_like(ins_mask)

        aug_mask = aug_mask[:, :, np.newaxis]

        image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).contiguous().float()
        aug_mask = torch.from_numpy(aug_mask).permute(2, 0, 1).contiguous().float()
        seg_mask = torch.from_numpy(seg_mask).contiguous().long()
        ins_mask = torch.from_numpy(ins_mask).contiguous().long()

        return image_patch, aug_mask, seg_mask, ins_mask

    def __getitem__(self, idx):
        image_patch, aug_mask, seg_mask, ins_mask = self.get_single_item(idx, 0)
        
        if self.norm != None:
            assert len(self.norm) == 2 
            mean = self.norm[0]
            std = self.norm[1]
            for t, m, s in zip(image_patch, mean, std):
                t.sub_(m).div_(s)
        
        return image_patch, aug_mask, seg_mask, ins_mask


class CocoPatchDataSet(PatchDataSet):
    def __init__(self, *args, **kwargs):
        super(CocoPatchDataSet, self).__init__('./data/coco', *args, **kwargs)


class VOCAugPatchDataSet(PatchDataSet):
    def __init__(self, *args, **kwargs):
        super(VOCAugPatchDataSet, self).__init__('./data/VOCAug', *args, **kwargs)


class VOC2012PatchDataSet(PatchDataSet):
    def __init__(self, *args, **kwargs):
        super(VOC2012PatchDataSet, self).__init__('./data/VOC2012', *args, **kwargs)

        
# add by sunpeng.
class YoutubePatchDataSet(Dataset):
    num_class = -1
    ignore_label = 255
    
    def readFlow(self, name):
        f = open(name, 'rb')
        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height,
                                                                       width, 2))
        f.close()
        return flow.astype(np.float32)

    def __init__(self, dataset_path='', gt_path='', flow_path='', data_list='', norm=None, is_train=True, sub_batch=1):
        random.seed()
        
        #data_list has 10 item in one line.
        self.img_path = dataset_path
        self.gt_path = gt_path
        self.flow_path = flow_path

        self.img_list = []
        lines = open(data_list ,"r").readlines()
        for line in lines:
            line = line.strip().split()
            self.img_list.append((line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9]))

        self.norm = norm
        self.patch_size = 54 * 8 + 1
        self.resize_w = 1280
        self.resize_h = 720
        self.is_train = is_train
        self.sub_batch = sub_batch
        
    def __len__(self):
        return len(self.img_list)
        
    def aug_mask(self, ins_mask):
        if np.count_nonzero(ins_mask == 1) == 0:
            ins_mask[0][0] = 1
        [y, x] = np.where(ins_mask == 1)
        wmax = max(x)
        wmin = min(x)
        hmax = max(y)
        hmin = min(y)
        _, contour, _ = cv2.findContours(ins_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour != []:
            contour = np.squeeze(np.vstack(contour))
        match_point = 5
        if (contour.shape[0] > match_point * 10):
            control_points = []
            for j in range(match_point):
                idx = random.randint(int(contour.shape[0] * (2 * j) / (2 * match_point)), int(contour.shape[0] * (2 * j + 1) / (2 * match_point)))
                sx, sy = contour[idx, :]
                tx = sx + random.uniform(-0.1 * (wmax - wmin + 1), 0.1 * (wmax - wmin + 1))
                ty = sy + random.uniform(-0.1 * (hmax - hmin + 1), 0.1 * (hmax - hmin + 1))
                control_points.append((sx, sy, tx, ty))

            try:
                t = from_control_points(control_points, backwards=True)
                ins_mask2 = ins_mask.copy()
                [x, y] = np.meshgrid(range(ins_mask.shape[1]), range(ins_mask.shape[0]))
                xy = np.vstack((x.flatten(), y.flatten()))
                # newxy = xy.copy()
                # for i in range(xy.shape[1]):
                newxy = np.round(np.array([t.transform(*xy[:, i]) for i in range(xy.shape[1])])).astype(int)
                newxy[:, 0] = np.clip(newxy[:, 0], 0, ins_mask.shape[1] - 1)
                newxy[:, 1] = np.clip(newxy[:, 1], 0, ins_mask.shape[0] - 1)
                ins_mask[xy[1, :], xy[0, :]] = ins_mask2[newxy[:, 1], newxy[:, 0]]
            except Exception:
                print('warp error!')

            patch_num = random.randint(1, 5)
            for i in range(patch_num):
                tot = np.count_nonzero(ins_mask)
                x = random.randint(0, ins_mask.shape[1] - 1)
                y = random.randint(0, ins_mask.shape[0] - 1)
                tot = int(tot * random.uniform(0.02, 0.05))
                key = (1 if ins_mask[y, x] == 0 else 0)
                for j in range(tot):
                    cv2.circle(ins_mask, (x, y), (2 if ins_mask[y, x] == 0 else 8), key, thickness=-1)
                    x = x + random.randint(-1, 1)
                    y = y + random.randint(-1, 1)
                    x = np.clip(x, 0, ins_mask.shape[1] - 1)
                    y = np.clip(y, 0, ins_mask.shape[0] - 1)

        dilation_size = random.randint(0, 2) * 2 + 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size * 2 + 1, dilation_size * 2 + 1), (dilation_size, dilation_size))
        ins_mask = cv2.dilate(ins_mask, element)
        return ins_mask

    def gen_bbox_mask(self, pre_image, pre_ins_label, flow, inverse_flow, pre_image_2, pre_ins_label_2, flow_2, inverse_flow_2, image, ins_label, idx):
        aug = 0.1
        enlage = random.uniform(0.1, 0.2)
        # pre image and label...
        [y, x] = np.where(pre_ins_label == idx)
        wmax = max(x) + 1
        wmin = min(x)
        hmax = max(y) + 1
        hmin = min(y)

        bbox_h = hmax - hmin
        bbox_w = wmax - wmin

        wmin += random.randint(-int(aug * bbox_w), int(aug * bbox_w))
        wmax += random.randint(-int(aug * bbox_w), int(aug * bbox_w))
        hmin += random.randint(-int(aug * bbox_h), int(aug * bbox_h))
        hmax += random.randint(-int(aug * bbox_h), int(aug * bbox_h))

        bbox_h = hmax - hmin
        bbox_w = wmax - wmin
        assert (bbox_h > 0)
        assert (bbox_w > 0)

        wmin = np.clip((wmin - enlage * bbox_w), 0, pre_ins_label.shape[1] - 1)
        wmax = np.clip((wmax + enlage * bbox_w), wmin + 1, pre_ins_label.shape[1])
        hmin = np.clip((hmin - enlage * bbox_h), 0, pre_ins_label.shape[0] - 1)
        hmax = np.clip((hmax + enlage * bbox_h), hmin + 1, pre_ins_label.shape[0])

        bbox = [int(wmin), int(hmin), int(wmax), int(hmax)]
        assert (bbox[3] > bbox[1])
        assert (bbox[2] > bbox[0])
        
        pre_img_patch = cv2.resize(pre_image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        pre_ins_mask = cv2.resize(pre_ins_label[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        
        flow_patch = cv2.resize(flow[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        inverse_flow_patch = cv2.resize(inverse_flow[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        
        flow_patch_2 = cv2.resize(flow_2[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        inverse_flow_patch_2 = cv2.resize(inverse_flow_2[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        
        pre_ins_mask[(pre_ins_mask != idx) & (pre_ins_mask != YoutubePatchDataSet.ignore_label)] = 0
        pre_ins_mask[(pre_ins_mask == idx)] = 1
        
        # current image and label...
        # if current image has no this obj, return 
        [y, x] = np.where(pre_ins_label_2 == idx)
        if len(x) > 2:
            pre_img_patch_2 = cv2.resize(pre_image_2[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            pre_ins_mask_2 = cv2.resize(pre_ins_label_2[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            pre_ins_mask_2[(pre_ins_mask_2 != idx) & (pre_ins_mask_2 != YoutubePatchDataSet.ignore_label)] = 0
            pre_ins_mask_2[(pre_ins_mask_2 == idx)] = 1
        else:
            pre_img_patch_2 = cv2.resize(pre_image_2[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            pre_ins_mask_2 = np.zeros_like(pre_ins_mask)
            
        [y, x] = np.where(ins_label == idx)
        if len(x) > 2:
            image_patch = cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            ins_mask = cv2.resize(ins_label[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            ins_mask[(ins_mask != idx) & (ins_mask != YoutubePatchDataSet.ignore_label)] = 0
            ins_mask[(ins_mask == idx)] = 1
        else:
            image_patch = cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            ins_mask = np.zeros_like(pre_ins_mask)

        return pre_img_patch, pre_ins_mask, flow_patch, inverse_flow_patch, pre_img_patch_2, pre_ins_mask_2, flow_patch_2, inverse_flow_patch_2, image_patch, ins_mask

    def get_single_item(self, idx, b_idx):
        # all scale is WIDTH=1280, WIDTH=720
        pre_image = cv2.imread(os.path.join(self.img_path, self.img_list[idx][0])).astype(np.float32)
        pre_ins_label = cv2.imread(os.path.join(self.gt_path, self.img_list[idx][1]), cv2.IMREAD_UNCHANGED)
        
        pre_image = cv2.resize(pre_image, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
        pre_ins_label = cv2.resize(pre_ins_label, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        
        flow = self.readFlow(os.path.join(self.flow_path, self.img_list[idx][2]))
        inverse_flow = self.readFlow(os.path.join(self.flow_path, self.img_list[idx][3]))
        
        # second pairs.
        pre_image_2 = cv2.imread(os.path.join(self.img_path, self.img_list[idx][4])).astype(np.float32)
        pre_ins_label_2 = cv2.imread(os.path.join(self.gt_path, self.img_list[idx][5]), cv2.IMREAD_UNCHANGED)
        
        pre_image_2 = cv2.resize(pre_image_2, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
        pre_ins_label_2 = cv2.resize(pre_ins_label_2, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        
        flow_2 = self.readFlow(os.path.join(self.flow_path, self.img_list[idx][6]))
        inverse_flow_2 = self.readFlow(os.path.join(self.flow_path, self.img_list[idx][7]))
        
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx][8])).astype(np.float32)
        ins_label = cv2.imread(os.path.join(self.gt_path, self.img_list[idx][9]), cv2.IMREAD_UNCHANGED)
        
        image = cv2.resize(image, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
        ins_label = cv2.resize(ins_label, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        
        # flip
        if (random.random() > 0.5 and self.is_train):
            image = np.fliplr(image)
            pre_image = np.fliplr(pre_image)
            pre_image_2 = np.fliplr(pre_image_2)
            ins_label = np.fliplr(ins_label)
            pre_ins_label = np.fliplr(pre_ins_label)
            pre_ins_label_2 = np.fliplr(pre_ins_label_2)
            # flow flip.
            flow[:,:,1] = np.fliplr(flow[:,:,1])
            flow[:,:,0] = -np.fliplr(flow[:,:,0])
            inverse_flow[:,:,1] = np.fliplr(inverse_flow[:,:,1])
            inverse_flow[:,:,0] = -np.fliplr(inverse_flow[:,:,0])
            
            flow_2[:,:,1] = np.fliplr(flow_2[:,:,1])
            flow_2[:,:,0] = -np.fliplr(flow_2[:,:,0])
            inverse_flow_2[:,:,1] = np.fliplr(inverse_flow_2[:,:,1])
            inverse_flow_2[:,:,0] = -np.fliplr(inverse_flow_2[:,:,0])
            
        # rotate
        if (random.random() > 0.5 and self.is_train):
            angl = random.uniform(-10, 10)
            h, w = image.shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, angl, 1.0)
            image = cv2.warpAffine(image, map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            pre_image = cv2.warpAffine(pre_image, map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            pre_image_2 = cv2.warpAffine(pre_image_2, map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            flow = cv2.warpAffine(flow, map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            inverse_flow = cv2.warpAffine(inverse_flow, map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            flow_2 = cv2.warpAffine(flow_2, map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            inverse_flow_2 = cv2.warpAffine(inverse_flow_2, map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            ins_label = cv2.warpAffine(ins_label, map_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=YoutubePatchDataSet.ignore_label)
            pre_ins_label = cv2.warpAffine(pre_ins_label, map_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=YoutubePatchDataSet.ignore_label)
            pre_ins_label_2 = cv2.warpAffine(pre_ins_label_2, map_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=YoutubePatchDataSet.ignore_label)

        idxs = np.unique(pre_ins_label)
        idxs = list(idxs[(idxs != 0) & (idxs != YoutubePatchDataSet.ignore_label)])

        if len(idxs) == 0:
#             print("No Instance!!!!!!!!!!!!!!!!!!")
            return self.get_single_item(random.randint(0, self.__len__() - 1), b_idx)
        else:
            idx = random.sample(idxs, 1)

        pre_img_patch, pre_ins_mask, flow_patch, inverse_flow_patch, pre_img_patch_2, pre_ins_mask_2, flow_patch_2, inverse_flow_patch_2, image_patch, ins_mask = self.gen_bbox_mask(pre_image, pre_ins_label, flow, inverse_flow, pre_image_2, pre_ins_label_2, flow_2, inverse_flow_2, image, ins_label, idx)
            
        # here, I change it....
        seed_number = random.random()
        # 40% augu-like, 60% real gt
        if (seed_number > 0.6):
            pre_aug_mask = self.aug_mask((pre_ins_mask == 1).astype(np.uint8))
        else :
            pre_aug_mask = pre_ins_mask

        pre_aug_mask = pre_aug_mask[:, :, np.newaxis]

        pre_image_patch = torch.from_numpy(pre_img_patch).permute(2, 0, 1).contiguous().float()
        pre_aug_mask = torch.from_numpy(pre_aug_mask).permute(2, 0, 1).contiguous().float()
        pre_ins_mask = torch.from_numpy(pre_ins_mask).contiguous().long()
        
        pre_image_patch_2 = torch.from_numpy(pre_img_patch_2).permute(2, 0, 1).contiguous().float()
        pre_ins_mask_2 = torch.from_numpy(pre_ins_mask_2).contiguous().long()
        
        flow_patch = torch.from_numpy(flow_patch).permute(2, 0, 1).contiguous().float()
        inverse_flow_patch = torch.from_numpy(inverse_flow_patch).permute(2, 0, 1).contiguous().float()
        
        flow_patch_2 = torch.from_numpy(flow_patch_2).permute(2, 0, 1).contiguous().float()
        inverse_flow_patch_2 = torch.from_numpy(inverse_flow_patch_2).permute(2, 0, 1).contiguous().float()
        
        image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).contiguous().float()
        ins_mask = torch.from_numpy(ins_mask).contiguous().long()

        return pre_image_patch, pre_aug_mask, pre_ins_mask, flow_patch, inverse_flow_patch, pre_image_patch_2, pre_ins_mask_2, flow_patch_2, inverse_flow_patch_2, image_patch, ins_mask
        
    def __getitem__(self, idx):
#         pre_image_patch, pre_aug_mask, pre_ins_mask, flow_patch, flow_patch_inverse, image_patch, ins_mask = self.get_single_item(idx, 0)
        pre_image_patch, pre_aug_mask, pre_ins_mask, flow_patch, inverse_flow_patch, pre_image_patch_2, pre_ins_mask_2, flow_patch_2, inverse_flow_patch_2, image_patch, ins_mask = self.get_single_item(idx, 0)
    
        if self.norm != None:
            assert len(self.norm) == 2
            mean = self.norm[0]
            std = self.norm[1]
            for t, m, s in zip(pre_image_patch, mean, std):
                t.sub_(m).div_(s)
            for z, x, y in zip(pre_image_patch_2, mean, std):
                z.sub_(x).div_(y)
            for q, w, e in zip(image_patch, mean, std):
                q.sub_(w).div_(e)
        return pre_image_patch, pre_aug_mask, pre_ins_mask, flow_patch, inverse_flow_patch, pre_image_patch_2, pre_ins_mask_2, flow_patch_2, inverse_flow_patch_2, image_patch, ins_mask
# over by sunpeng.


if __name__ == "__main__":
        dataset = YoutubePatchDataSet(dataset_path='', gt_path='',data_list='/mnt/lustre/share/sunpeng/video-seg-workshop/0801_warp/three_pairs_shuffle.txt',norm=None)
        temp = 0
        while(True):
            temp += 1
            pre_image_patch, pre_aug_mask, pre_ins_mask, flow_patch, inverse_flow_patch, pre_image_patch_2, pre_ins_mask_2, flow_patch_2, inverse_flow_patch_2, image_patch, ins_mask = dataset.get_single_item(random.randint(0, dataset.__len__() - 1), 0)
#             pre_image_patch_2
#             pre_aug_mask = pre_aug_mask.cpu().numpy().astype(np.uint8).transpose((1,2,0))
            pre_image_patch_2 = pre_image_patch_2.cpu().numpy().astype(np.uint8).transpose((1,2,0))
#             print (pre_image_patch_2.shape)

            cv2.imwrite('temp/'+str(temp)+'pre_image.png', pre_image_patch_2)
        
            image_patch = image_patch.cpu().numpy().astype(np.uint8).transpose((1,2,0))
            cv2.imwrite('temp/'+str(temp)+'_image.png', image_patch)
            
            pre = pre_ins_mask_2.cpu().numpy().astype(np.uint8)
            ins_mask = ins_mask.cpu().numpy().astype(np.uint8)
            print (np.unique(pre))
            print ("pre")
            print (np.unique(ins_mask))
            print ("ins_mask")
            cv2.imwrite('temp/'+str(temp)+'pre_mask.png', pre * 200)
            cv2.imwrite('temp/'+str(temp)+'now_mask.png', ins_mask * 200)
            print (temp)
            
            
#             cv2.imwrite('temp/'+str(temp)+'image.png', image)
#             cv2.imwrite('temp/'+str(temp)+'image.png', image)
#             cv2.imwrite('temp/'+str(temp)+'image.png', image)
