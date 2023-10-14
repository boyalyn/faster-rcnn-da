import os

import ipdb
import matplotlib
from tqdm import tqdm
import torch
from Utils.config import opt
# from data.dataset import Dataset, TestDataset, inverse_normalize
from data.jacs_dataset import JACSDatasetBase
from model import DANN
from torch.utils import data as data_
from dann_trainer import FasterRCNNTrainer
from Utils import array_tool as at
from Utils.vis_tool import visdom_bbox
from Utils.eval_tool import eval_detection_voc

from Configs.Config import Config


def eval(dataloader, faster_rcnn, test_num=10000):
    print("predicting...")
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, gt_bboxes_, gt_labels_, _) in tqdm(enumerate(dataloader)):
        imgs, gt_bboxes_, gt_labels_ = imgs.to(opt.device).float(), gt_bboxes_.to(opt.device), gt_labels_.to(opt.device)
        print("gt_bboxes_:", gt_bboxes_.shape)
        print("gt_labels_:", gt_labels_.shape)
        gt_difficults = None
        sizes = imgs.shape
        sizes = [sizes[2], sizes[3]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.cpu().numpy())
        gt_labels += list(gt_labels_.cpu().numpy())
        # gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    print("evaluating...")
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=False)
    return result


def train(opt):

    print('load data #################')
    src_dataset = JACSDatasetBase(root=opt.train_root,
                              anno_file=opt.train_anno_file)
    
    src_dataloader = data_.DataLoader(src_dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    
    dst_dataset = JACSDatasetBase(root=opt.test_root,
                              anno_file=opt.test_anno_file)
    dst_dataloader = data_.DataLoader( dst_dataset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    # model
    faster_rcnn = DANN().to(opt.device)
    print('model construct completed')


    trainer = FasterRCNNTrainer(faster_rcnn).to(opt.device)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    # lr_ = opt.lr
    src_iterator = iter(src_dataloader)
    dst_iterator = iter(dst_dataloader)

    for ii in range(opt.iter):

        print(f"iteration {ii}:\n")
        best_map = 0.0

        try:
            img, bbox_, label_, scale = next(src_iterator)
        except:
            # print("source iterator ended")
            src_iterator = iter(src_dataloader)
            img, bbox_, label_, scale = next(src_iterator)

        scale = at.scalar(scale)
        img, bbox, label = img.to(opt.device).float(), bbox_.to(opt.device), label_.to(opt.device)
        print("train source")
        trainer.train_step(img, bbox, label, scale, "source")

        try:
            img, bbox_, label_, scale = next(dst_iterator)
        except:
            print("target iterator ended")
            dst_iterator = iter(dst_dataloader)
            img, bbox_, label_, scale = next(dst_iterator)

        scale = at.scalar(scale)
        img, bbox, label = img.to(opt.device).float(), bbox_.to(opt.device), label_.to(opt.device)
        print("train target")
        trainer.train_step(img, bbox, label, scale, "target")


        # if ii != 0 and ii % opt.interval == 0:
        if ii%opt.interval == 0:
            print(trainer.get_meter_data())
            trainer.reset_meters()

            # eval train
            eval_result_train = eval(src_dataloader, faster_rcnn, test_num=1)
            # trainer.vis.plot('test_map', eval_result['map'])
            # lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
            log_info_train = f"train metrics: {eval_result_train}"
            # trainer.vis.log(log_info)
            print(log_info_train)

            # eval test
            eval_result_test = eval(dst_dataloader, faster_rcnn, test_num=1)
            # trainer.vis.plot('test_map', eval_result['map'])
            # lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
            log_info_test = f"test metrics: {eval_result_test}"
            # trainer.vis.log(log_info)
            print(log_info_test)

            if eval_result_test['map'] > best_map:
                best_map = eval_result_test['map']
                best_path = trainer.save(best_map=best_map)
            

if __name__ == "__main__":

    opt = Config().get_opt()

    train(opt)