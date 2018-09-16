import os
import time
import logging
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils2 import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, IterLRScheduler, \
    DistributedGivenIterationSampler, simple_group_split
from torchE.D import dist_init, average_gradients, DistModule
from tensorboardX import SummaryWriter

from speed_utils.bbox import label_to_prob, prob_to_label, combine_prob, gen_bbox, IoU
from speed_utils import flow as flo
from speed_utils.disp import labelcolormap

# import segdata as datasets
import patch_dataset as datasets
import segtransforms as transforms
# from pspnet import PSPNet
from utils import AverageMeter, poly_learning_rate, intersectionAndUnion


# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--data_root', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012', help='data root')
    parser.add_argument('--train_list', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/train.txt',
                        help='train list')
    parser.add_argument('--val_list', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/val.txt',
                        help='val list')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone network type')
    parser.add_argument('--net_type', type=int, default=0, help='0-single branch, 1-div4 branch')
    parser.add_argument('--layers', type=int, default=50, help='layers number of based resnet')
    parser.add_argument('--syncbn', type=int, default=1, help='adopt syncbn or not')
    parser.add_argument('--classes', type=int, default=21, help='number of classes')
    parser.add_argument('--crop_h', type=int, default=473, help='train crop size h')
    parser.add_argument('--crop_w', type=int, default=473, help='train crop size w')
    parser.add_argument('--scale_min', type=float, default=0.5, help='minimum random scale')
    parser.add_argument('--scale_max', type=float, default=2.0, help='maximum random scale')
    parser.add_argument('--rotate_min', type=float, default=-10, help='minimum random rotate')
    parser.add_argument('--rotate_max', type=float, default=10, help='maximum random rotate')
    parser.add_argument('--zoom_factor', type=int, default=1, help='zoom factor in final prediction map')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label in ground truth')
    parser.add_argument('--ins_weight', type=float, default=5, help='loss weight for aux branch')

    parser.add_argument('--gpu', type=int, default=[0, 1, 2, 3], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--bn_group', type=int, default=16, help='group number for sync bn')
    parser.add_argument('--batch_size_val', type=int, default=1,
                        help='batch size for validation during training, memory and speed tradeoff')
    parser.add_argument('--base_lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--power', type=float, default=0.9, help='power in poly learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--save_step', type=int, default=10, help='model save step (default: 10)')
    parser.add_argument('--save_path', type=str, default='tmp', help='model and summary save path')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--weight', type=str, default='', help='path to weight (default: none)')
    parser.add_argument('--weight_flow', type=str, default='', help='path to weight_flow (default: none)')
    parser.add_argument('--evaluate', type=int, default=0,
                        help='evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend')
    parser.add_argument('--port', default='23456', type=str)
    parser.add_argument('--dataset_name', default='youtube', type=str, help='coco or voc or youtube')
    return parser




def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def change_prefix(state_dict, prefix, replace_prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    logger.info('replace prefix \'{}\''.format(prefix))
    f = lambda x: x.replace(prefix, replace_prefix) if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def load_state(path, model, optimizer=None):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        logger.info("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(path))


def main():
    global args, logger, writer
    args = get_parser().parse_args()
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    rank, world_size = dist_init(args.port)
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    # if len(args.gpu) == 1:
    #   args.syncbn = False
    if rank == 0:
        logger.info(args)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.crop_h - 1) % 8 == 0 and (args.crop_w - 1) % 8 == 0
    assert args.net_type in [0, 1, 2, 3]

    if args.bn_group == 1:
        args.bn_group_comm = None
    else:
        assert world_size % args.bn_group == 0
        args.bn_group_comm = simple_group_split(world_size, rank, world_size // args.bn_group)

    if rank == 0:
        logger.info("=> creating two branch model ...")
        logger.info("Classes: {}".format(args.classes))

    if args.net_type == 0:
        #         from pspnet import PSPNet
        from pspnet2S import PSP2S
        model = PSP2S(backbone=args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor,
                      syncbn=args.syncbn, group_size=args.bn_group, group=args.bn_group_comm).cuda()
    logger.info(model)
    # optimizer = torch.optim.SGD(model.parameters(), args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # newly introduced layer with lr x10
    optimizer = torch.optim.SGD(
        [{'params': model.rgb_branch.layer0_img.parameters()},
         {'params': model.rgb_branch.layer0_ins.parameters()},
         {'params': model.rgb_branch.layer0_conv1.parameters()},
         {'params': model.rgb_branch.layer1.parameters()},
         {'params': model.rgb_branch.layer2.parameters()},
         {'params': model.rgb_branch.layer3.parameters()},
         {'params': model.rgb_branch.layer4.parameters()},
         {'params': model.rgb_branch.conv6.parameters()},
         {'params': model.rgb_branch.conv1_1x1.parameters()},
         {'params': model.rgb_branch.ppm.parameters(), 'lr': args.base_lr * 10},
         {'params': model.rgb_branch.cls_ins.parameters(), 'lr': args.base_lr * 10},

         {'params': model.flow_branch.layer0_flo.parameters()},
         {'params': model.flow_branch.layer0_ins.parameters()},
         {'params': model.flow_branch.layer0_conv1.parameters()},
         {'params': model.flow_branch.layer1.parameters()},
         {'params': model.flow_branch.layer2.parameters()},
         {'params': model.flow_branch.layer3.parameters()},
         {'params': model.flow_branch.layer4.parameters()},
         {'params': model.flow_branch.conv6.parameters()},
         {'params': model.flow_branch.conv1_1x1.parameters()},
         {'params': model.flow_branch.ppm.parameters(), 'lr': args.base_lr * 10},
         {'params': model.flow_branch.cls_ins.parameters(), 'lr': args.base_lr * 10}],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.enabled = True
    cudnn.benchmark = True
    criterion = nn.NLLLoss(ignore_index=args.ignore_label).cuda()

    if args.weight:
        def map_func(storage, location):
            return storage.cuda()

        if os.path.isfile(args.weight):
            # load rgb branch params.
            logger.info("=> loading rgb weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location=map_func)
            new_dict = remove_prefix(checkpoint['state_dict'], 'module.')
            model.rgb_branch.load_state_dict(new_dict, strict=True)
            logger.info("=> loaded rgb weight '{}'".format(args.weight))

            # load flow branch params.
            logger.info("=> loading flow weight '{}'".format(args.weight_flow))
            checkpoint_flow = torch.load(args.weight_flow, map_location=map_func)
            new_dict_flow = remove_prefix(checkpoint_flow['state_dict'], 'module.')
            model.flow_branch.load_state_dict(new_dict_flow, strict=True)
            logger.info("=> loaded flow weight '{}'".format(args.weight_flow))

    else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    model = DistModule(model)

    if args.resume:
        load_state(args.resume, model, optimizer)

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if args.dataset_name == 'coco':
        logger.info("=> load coco patch datasets....")
        train_data = datasets.PatchDataSet(dataset_path='/mnt/lustre/share/sunpeng/voc_coco_label/coco/',
                                           data_list='train2017', norm=[mean, std])
    elif args.dataset_name == 'voc':
        logger.info("=> load voc patch datasets....")
        train_data = datasets.PatchDataSet(dataset_path='/mnt/lustre/share/sunpeng/voc_coco_label/voc/',
                                           data_list='train_val_ins', norm=[mean, std])
    elif args.dataset_name == 'youtube':
        logger.info("=> load youtube shuffle three pairs patch datasets....")
        train_data = datasets.YoutubePatchDataSet(dataset_path='', gt_path='',
                                                  data_list='/mnt/lustre/sunpeng/Research/video-seg-workshop/models/deeplabv3_models/final_2s_seg_youtube_online_two_finetune_first_frame/add_first_frame_final_0823.txt',
                                                  norm=[mean, std])
    elif args.dataset_name == 'youtube_and_davis':
        logger.info("=> load youtube and davis shuffle three pairs patch datasets....")
        train_data = datasets.YoutubePatchDataSet(dataset_path='', gt_path='',                                                                                                                                           data_list='/mnt/lustre/share/sunpeng/video-seg-workshop/davis/DAVIS/train_shuffle_davis.txt', norm=[mean, std])
    else:
        logger.info("=> no right datset name found, please input coco or voc.")
        assert 0 == 1
    # train_data = datasets.SegData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    train_sampler = DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    if args.evaluate:
        val_transform = transforms.Compose([
            transforms.Crop([args.crop_h, args.crop_w], crop_type='center', padding=mean,
                            ignore_label=args.ignore_label),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        val_data = datasets.SegData(split='val', data_root=args.data_root, data_list=args.val_list,
                                    transform=val_transform)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs + 1):
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch,
                                                                 args.zoom_factor, args.batch_size, args.ins_weight)
        if rank == 0:
            writer.add_scalar('loss_train', loss_train, epoch)
            writer.add_scalar('mIoU_train', mIoU_train, epoch)
            writer.add_scalar('mAcc_train', mAcc_train, epoch)
            writer.add_scalar('allAcc_train', allAcc_train, epoch)
        # write parameters histogram costs lots of time
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)
        if epoch % args.save_step == 0 and rank == 0:
            filename = args.save_path + '/train_epoch_' + str(epoch) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       filename)
            # if epoch / args.save_step > 2:
            #    deletename = args.save_path + '/train_epoch_' + str(epoch - args.save_step*2) + '.pth'
            #    os.remove(deletename)
        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion, args.classes,
                                                                args.zoom_factor)
            writer.add_scalar('loss_val', loss_val, epoch)
            writer.add_scalar('mIoU_val', mIoU_val, epoch)
            writer.add_scalar('mAcc_val', mAcc_val, epoch)
            writer.add_scalar('allAcc_val', allAcc_val, epoch)


def train(train_loader, model, criterion, optimizer, epoch, zoom_factor, batch_size, ins_weight):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    # aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    end = time.time()

    for i, (pre_image_patch, pre_aug_mask, pre_ins_mask, flow_patch, inverse_flow_patch, pre_image_patch_2, pre_ins_mask_2, flow_patch_2, inverse_flow_patch_2, image_patch, ins_mask) in enumerate(train_loader):
        
        #abandon the two objs.
        pre_image_patch = None
        pre_ins_mask = None
        
        data_time.update(time.time() - end)
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = args.epochs * len(train_loader)
        if args.net_type == 0:
            index_split = 4
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=index_split)
        # pre image1.
#         print (np.squeeze(pre_aug_mask.cpu().numpy()).shape) # (1, 1, 433, 433)
        pre_tmp_out = np.squeeze(pre_aug_mask.cpu().numpy(), axis=0).transpose((1, 2, 0))
        pre_tmp_out = np.squeeze(label_to_prob(pre_tmp_out, 1))
        
        flow_patch = np.squeeze(flow_patch.cpu().numpy()).transpose((1, 2, 0))
        inverse_flow_patch_numpy = np.squeeze(inverse_flow_patch.cpu().numpy()).transpose((1, 2, 0))
        warp_pred = prob_to_label(flo.get_warp_label(flow_patch, inverse_flow_patch_numpy, pre_tmp_out))
        pre = torch.from_numpy(warp_pred).contiguous().float()
        warped_pred_aug_mask_var = torch.autograd.Variable(torch.unsqueeze(torch.unsqueeze(pre, dim=0), dim=0).cuda(async=True))
        
        inverse_flow_patch_var = torch.autograd.Variable(inverse_flow_patch.cuda(async=True))
        
        pre_input_var = torch.autograd.Variable(pre_image_patch_2.cuda(async=True))
        # input model
        pre_output_ins = model(pre_input_var, warped_pred_aug_mask_var, inverse_flow_patch_var)

        seg_loss = 0
        pre_ins_mask222 = torch.autograd.Variable(pre_ins_mask_2).squeeze(1).long()
        pre_ins_mask_var = torch.autograd.Variable(pre_ins_mask222.cuda(async=True))
#         #debug1
#         import cv2
# #         pre_image_patch_2_debug = np.squeeze(pre_image_patch_2.cpu().numpy()).transpose((1, 2, 0))
#         warped_pred_aug_mask_var_debug = torch.unsqueeze(pre, dim=0).cpu().numpy().transpose((1, 2, 0))
#         pre_ins_mask_var_debug = np.squeeze(pre_ins_mask_2.cpu().numpy())
        
#         warped_pred_aug_mask_var_debug[warped_pred_aug_mask_var_debug == 255] = 0
#         pre_ins_mask_var_debug[pre_ins_mask_var_debug == 255] = 0
# #         cv2.imwrite("debug/"+str(rank)+ "__" + str(i) +"pre_patch.jpg", pre_image_patch_2_debug)
#         cv2.imwrite("debug/"+str(rank)+ "__" + str(i) +"pre_warped_patch.png", warped_pred_aug_mask_var_debug * 255)
#         cv2.imwrite("debug/"+str(rank)+ "__" + str(i) +"pre_ins_mask.png", pre_ins_mask_var_debug * 255)
#         #debug1 over
        ins_loss = criterion(pre_output_ins, pre_ins_mask_var) / world_size
        last_ins_loss = ins_loss
        loss = seg_loss + ins_loss
        loss = loss * 0.8 # tow loss weight.
        
        # current image.
        image_patch_var = torch.autograd.Variable(image_patch.cuda(async=True))
        # pre_output_ins  (1, 2 ,433, 433)
#         pre_output_ins_var = torch.argmax(pre_output_ins, dim=1, keepdim=True).float()
        pre_output_ins_var = torch.argmax(pre_output_ins[0], dim=0, keepdim=True).float()
#         tmp_out = pre_output_ins_var.data.max(1)[1].cpu().numpy().transpose((1,2,0))
        tmp_out = pre_output_ins_var.cpu().numpy().transpose((1,2,0))
#         print (np.unique(tmp_out))
#         cv2.imwrite("debug/"+str(rank)+ "__" + str(i) +"pre_out.png", tmp_out * 255)
        tmp_prob = np.squeeze(label_to_prob(tmp_out, 1))
        
        flow_patch_2 = np.squeeze(flow_patch_2.cpu().numpy()).transpose((1, 2, 0))
        inverse_flow_patch_2_numpy = np.squeeze(inverse_flow_patch_2.cpu().numpy()).transpose((1, 2, 0))
        #warp
        warp_pred = prob_to_label(flo.get_warp_label(flow_patch_2, inverse_flow_patch_2_numpy, tmp_prob))
        pre = torch.from_numpy(warp_pred).contiguous().float()
        warped_pred_aug_mask_var = torch.autograd.Variable(torch.unsqueeze(torch.unsqueeze(pre, dim=0), dim=0).cuda(async=True))
        # pred_aug_mask_var warp to this image.
        inverse_flow_patch_var_2 = torch.autograd.Variable(inverse_flow_patch_2.cuda(async=True))
        
        # input model
        output_ins = model(image_patch_var, warped_pred_aug_mask_var, inverse_flow_patch_var_2)

        ins_mask = torch.autograd.Variable(ins_mask).squeeze(1).long()
        ins_mask_var = torch.autograd.Variable(ins_mask.cuda(async=True))
        #debug2
#         pre_image_patch_2_debug = np.squeeze(image_patch.cpu().numpy()).transpose((1, 2, 0))
#         warped_pred_aug_mask_var_debug = torch.unsqueeze(pre, dim=0).cpu().numpy().transpose((1, 2, 0))
#         pre_ins_mask_var_debug = np.squeeze(ins_mask.cpu().numpy())
#         warped_pred_aug_mask_var_debug[warped_pred_aug_mask_var_debug == 255] = 0
#         pre_ins_mask_var_debug[pre_ins_mask_var_debug == 255] = 0
# #         cv2.imwrite("debug/"+str(rank)+ "__" + str(i) +"now_patch.jpg", pre_image_patch_2_debug)
#         cv2.imwrite("debug/"+str(rank)+ "__" + str(i) +"now_warped_patch.png", warped_pred_aug_mask_var_debug * 255)
#         cv2.imwrite("debug/"+str(rank)+ "__" + str(i) +"now_ins_mask.png", pre_ins_mask_var_debug * 255)
#         #debug2 over
        seg_loss = 0
        ins_loss = criterion(output_ins, ins_mask_var) / world_size
        loss = ins_loss + seg_loss

        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()

        output = output_ins.data.max(1)[1].cpu().numpy()
        target = ins_mask.cpu().numpy()
        intersection, union, target = intersectionAndUnion(output, target, 2, args.ignore_label)  # 1 = args.classes
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        reduced_loss = last_ins_loss.data.clone()
        reduced_main_loss = ins_loss.data.clone()
        # reduced_aux_loss = ins_loss.data.clone()  # ins_loss replace here.
        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_main_loss)
        # dist.all_reduce(reduced_aux_loss)

        main_loss_meter.update(reduced_main_loss[0], image_patch.size(0))
        # aux_loss_meter.update(reduced_aux_loss[0], input.size(0))
        loss_meter.update(reduced_loss[0], image_patch.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if rank == 0:
            if (i + 1) % args.print_freq == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'cur ins Loss {main_loss_meter.val:.4f} '
                            'pre ins Loss {loss_meter.val:.4f} '
                            'Accuracy {accuracy:.4f}.'.format(epoch, args.epochs, i + 1, len(train_loader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              remain_time=remain_time,
                                                              main_loss_meter=main_loss_meter,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if rank == 0:
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion, classes, zoom_factor):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        target = target.cuda()  # non_blocking=True)
        output = model(input)
        if zoom_factor != 8:
            output = F.upsample(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        output = output.data.max(1)[1].cpu().numpy()
        target = target.cpu().numpy()
        intersection, union, target = intersectionAndUnion(output, target, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    for i in range(classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()
