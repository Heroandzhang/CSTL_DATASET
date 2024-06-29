import math
import sys
import time

import torch

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    result_csv = []
    target_csv = []

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time


        # print("targets:")
        # print(targets)
        # print("wwwwwwww")
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # print("result:")
        # print(res)
        # print("finished!!!!!!")

        # 生成csv前步骤，以便计算FROC
        for t in targets:
            # print(t)
            for j in range(len(t['boxes'])):
                target_data = []
                xmin = round(t['boxes'][j][0].item(), 4)
                ymin = round(t['boxes'][j][1].item(), 4)
                xmax = round(t['boxes'][j][2].item(), 4)
                ymax = round(t['boxes'][j][3].item(), 4)
                target_data.append(t['image_id'][j].item())
                target_data.append(xmin)
                target_data.append(ymin)
                target_data.append(xmax)
                target_data.append(ymax)
                target_data.append(round(xmin + ((xmax - xmin) / 2.0), 4))  # x_center
                target_data.append(round(ymin + ((ymax - ymin) / 2.0), 4))  # y_center
                target_data.append(((xmax - xmin) + (ymax - ymin)) / 2.0)  # 直径
                target_data.append(t['labels'][j].item())
                target_csv.append(target_data)


        for k in res:
            # print(k)
            for i in range(len(res[k]['boxes'])):
                # print(k)  # file_name
                # print(res[k]['boxes'][i])  # predict_boxes
                # print(res[k]['labels'][i])  # predict_labels
                # print(res[k]['scores'][i])  # predict_scores
                result_data = []
                xmin = round(res[k]['boxes'][i][0].item(), 4)  # round(x,4) 保留小数点后四位
                ymin = round(res[k]['boxes'][i][1].item(), 4)
                xmax = round(res[k]['boxes'][i][2].item(), 4)
                ymax = round(res[k]['boxes'][i][3].item(), 4)
                result_data.append(k)
                result_data.append(xmin)
                result_data.append(ymin)
                result_data.append(xmax)
                result_data.append(ymax)
                result_data.append(round(xmin + ((xmax - xmin) / 2.0), 4))  # x_center
                result_data.append(round(ymin + ((ymax - ymin) / 2.0), 4))  # y_center
                result_data.append(res[k]['labels'][i].item())
                result_data.append(res[k]['scores'][i].item())
                result_csv.append(result_data)


        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    print("target_csv:::")
    print(target_csv)
    print("result_csv:::")
    print(result_csv)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info, target_csv, result_csv


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
