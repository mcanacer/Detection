import argparse
import wandb

from torchvision import transforms

from dataset import CocoDataset
from backbone import MobileNetBackbone
from fpn import FPN
from towers import ConvolutionalTower
from heads import ConvolutionalHead
from assigners import FCOSAssigner
from point_generator import PointsGenerator
from box_coders import LTRB
from fcos import FeatureExtractor, FCOS
from evaluator import COCOEvaluator

import torch


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-dataset-dir', type=str, required=True)
    parser.add_argument('--train-dataset-json-dir', type=str, required=True)
    parser.add_argument('--eval-dataset-dir', type=str, required=True)
    parser.add_argument('--eval-dataset-json-dir', type=str, required=True)

    parser.add_argument('--train-dataset', type=str, required=True)
    parser.add_argument('--eval-dataset', type=str, required=True)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=1)

    parser.add_argument('--max-num-instances', type=int, default=40)

    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)

    parser.add_argument('--init-prob', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--decay-factor', type=float, default=0.1)

    parser.add_argument('--num-classes', type=int, default=90)
    parser.add_argument('--iou-threshold', type=float, default=0.5)

    parser.add_argument('--min-level', type=int, default=3)
    parser.add_argument('--max-level', type=int, default=7)
    parser.add_argument('--filter-size', type=int, default=128)

    parser.add_argument('--num-repeats', type=int, default=1)

    parser.add_argument('--initial-learning-rate', type=float, default=0.01)

    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--id', type=str)

    parser.add_argument('--step-checkpoint-path', type=str)
    parser.add_argument('--epoch-checkpoint-path', type=str)

    parser.add_argument('--eval-score-threshold', type=float, default=0.05)

    return parser.parse_args(args)


def everything(args):
    args = parse_args(args)

    train_dataset = CocoDataset(
        args.train_dataset_dir,
        args.train_dataset_json_dir,
        args.max_num_instances,
        transforms.Compose([transforms.Resize(args.height),
        transforms.CenterCrop(args.height),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
    )

    eval_dataset = CocoDataset(
        args.eval_dataset_dir,
        args.eval_dataset_json_dir,
        args.max_num_instances,
        transforms.Compose([transforms.Resize(args.height),
        transforms.CenterCrop(args.height),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, pin_memory=True)

    backbone = MobileNetBackbone()
    fpn = FPN(backbone.out_channels, args.filter_size)

    feature_map_indexes = [i for i in range(args.min_level, args.max_level + 1)]

    box_tower = ConvolutionalTower(args.num_repeats, args.filter_size)
    class_tower = ConvolutionalTower(args.num_repeats, args.filter_size)

    box_head = ConvolutionalHead(args.num_repeats, args.filter_size, out_filters=4)
    class_head = ConvolutionalHead(args.num_repeats, args.filter_size, out_filters=args.num_classes)
    centerness_head = ConvolutionalHead(args.num_repeats, args.filter_size, out_filters=1)

    points_generator = PointsGenerator()
    box_coder = LTRB()

    ranges = {
        3: (0, 64),
        4: (64, 128),
        5: (128, 256),
        6: (256, 512),
        7: (512, 1e8),
    }

    assigner = FCOSAssigner(ranges)

    model = FeatureExtractor(backbone, fpn, box_tower, class_tower, box_head, class_head, centerness_head, feature_map_indexes)

    detector = FCOS(
        model,
        args.num_classes,
        points_generator,
        assigner,
        box_coder,
        max_detections=args.max_num_instances,
        iou_threshold=args.iou_threshold,
    )

    optimizer = torch.optim.SGD(detector.parameters(), lr=args.initial_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    milestones = [60000, 80000]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.decay_factor)

    evaluator = COCOEvaluator(score_threshold=args.eval_score_threshold)

    run = wandb.init(
        project=args.project,
        name=args.project,
        reinit=True,
        config=vars(args)
    )

    return {
        'train_loader': train_loader,
        'eval_loader': eval_loader,
        'evaluator': evaluator,
        'detector': detector,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'run': run,
        'step_checkpoint_path': args.step_checkpoint_path,
        'epoch_checkpoint_path': args.epoch_checkpoint_path,
        'height': args.height,
        'widht': args.width
    }


