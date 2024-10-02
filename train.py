import torch
from dataset import VOC2007DetectionTiny
from backbone import *
from fpn import FPN
from retinanet import FeatureExtractor, RetinaNet
from heads import ConvolutionalHead
from anchor_generators import MultipleGridAnchor
from assigners import ArgmaxAssigner
from similarity_calculators import IouSimilarity
from box_coders import OffsetBoxCoder
import numpy as np

num_classes = 20
num_anchors = 9

dataset_dir = "/Users/muhammetcan/Desktop/RetinaNet"

train_dataset = VOC2007DetectionTiny(dataset_dir, "train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, pin_memory=True)

backbone = ResNet18()
fpn = FPN(backbone.out_channels, filter_size=256)

feature_map_indexes = [i for i in range(3, 8)]

box_head = ConvolutionalHead(num_repeats=4, filter_size=256, out_filters=num_anchors*4)
class_head = ConvolutionalHead(num_repeats=4, filter_size=256, out_filters=num_anchors*num_classes)

scales = [2 ** (i/3) for i in range(3)]
aspect_ratios = [0.5, 1, 2]

anchor_generator = MultipleGridAnchor(scales, aspect_ratios)

iou_similarity = IouSimilarity()

assigner = ArgmaxAssigner(iou_similarity, matched_threshold=0.5, unmatched_threshold=0.4, num_classes=num_classes)

box_coder = OffsetBoxCoder()

model = FeatureExtractor(backbone, fpn, box_head, class_head, feature_map_indexes)
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
        optimizer.zero_grad()
        loss = box_loss.mean() + class_loss.mean()
        print(loss)
        loss.backward()
        optimizer.step()
        epoch_loss.append(float(loss))
        total_loss = np.mean(epoch_loss)

    steps += 1

    scheduler.step(np.mean(epoch_loss))

