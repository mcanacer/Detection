import torch
from dataset import VOC2007DetectionTiny, CocoDataset
from backbone import *
from fpn import FPN
from retinanet import FeatureExtractor, RetinaNet
from towers import ConvolutionalTower
from heads import ConvolutionalHead
from anchor_generators import MultipleGridAnchor
from assigners import ArgmaxAssigner
from similarity_calculators import IouSimilarity
from box_coders import OffsetBoxCoder
import numpy as np

num_classes = 90
num_anchors = 9

dataset_dir = "/Users/muhammetcan/Desktop/Detection/val2017"

train_dataset = CocoDataset(dataset_dir, "/Users/muhammetcan/Desktop/Detection/annotations/instances_val2017.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, pin_memory=True)

backbone = ResNet50()
fpn = FPN(backbone.out_channels, filter_size=512)

feature_map_indexes = [i for i in range(3, 8)]

box_tower = ConvolutionalTower(num_repeats=3, filter_size=512, survival_prob=0.5)
class_tower = ConvolutionalTower(num_repeats=3, filter_size=512, survival_prob=0.5)

box_head = ConvolutionalHead(num_repeats=3, filter_size=512, out_filters=num_anchors*4, survival_prob=0.5)
class_head = ConvolutionalHead(num_repeats=3, filter_size=512, out_filters=num_anchors*num_classes, survival_prob=0.5)

scales = [2 ** (i/3) for i in range(3)]
aspect_ratios = [0.5, 1, 2]

anchor_generator = MultipleGridAnchor(scales, aspect_ratios)

iou_similarity = IouSimilarity()

assigner = ArgmaxAssigner(iou_similarity, matched_threshold=0.5, unmatched_threshold=0.4, num_classes=num_classes)

box_coder = OffsetBoxCoder()

model = FeatureExtractor(backbone, fpn, box_tower, class_tower, box_head, class_head, feature_map_indexes)
detector = RetinaNet(model, num_classes, anchor_generator, assigner, box_coder)

optimizer = torch.optim.SGD(detector.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

steps = 0

while True:
    detector.train()

    curr_lr = float(optimizer.param_groups[0]['lr'])

    epoch_loss = []

    for inputs in train_loader:
        box_loss, class_loss = detector(inputs)
        print(box_loss.mean())
        print(class_loss.mean())
        optimizer.zero_grad()
        loss = box_loss.mean() + class_loss.mean()
        print(loss)
        loss.backward()
        optimizer.step()
        epoch_loss.append(float(loss))
        total_loss = np.mean(epoch_loss)

    steps += 1

    scheduler.step(np.mean(epoch_loss))

