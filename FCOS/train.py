import torch
from dataset import VOC2007DetectionTiny
from backbone import *
from fpn import FPN
from fcos import FeatureExtractor, FCOS
from towers import ConvolutionalTower
from heads import ConvolutionalHead
from location_generator import LocationGenerator
from assigners import FCOSAssigner
from box_coders import LTRB
import numpy as np

num_classes = 20

dataset_dir = "/Users/muhammetcan/Desktop/RetinaNet"

train_dataset = VOC2007DetectionTiny(dataset_dir, 'train', image_size=224)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, pin_memory=True)

backbone = ResNet50()
fpn = FPN(backbone.out_channels, filter_size=256)

feature_map_indexes = [i for i in range(3, 8)]

box_tower = ConvolutionalTower(num_repeats=4, filter_size=256, survival_prob=0.5)

class_tower = ConvolutionalTower(num_repeats=4, filter_size=256, survival_prob=0.5)

box_head = ConvolutionalHead(num_repeats=4, filter_size=256, out_filters=4, survival_prob=0.5)

class_head = ConvolutionalHead(num_repeats=4, filter_size=256, out_filters=num_classes, survival_prob=0.5)

centerness_head = ConvolutionalHead(num_repeats=4, filter_size=256, out_filters=1, survival_prob=0.5)

location_generator = LocationGenerator()
box_coder = LTRB()

ranges = {
    3: (0, 64),
    4: (64, 128),
    5: (128, 256),
    6: (256, 512),
    7: (512, 1e8),
}

assigner = FCOSAssigner(box_coder, ranges)

model = FeatureExtractor(backbone, fpn, class_tower, box_tower, class_head, box_head, centerness_head, feature_map_indexes)

detector = FCOS(model, num_classes, location_generator, assigner, box_coder)

optimizer = torch.optim.SGD(detector.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

steps = 0

while True:
    detector.train()

    curr_lr = float(optimizer.param_groups[0]['lr'])

    epoch_loss = []

    for inputs in train_loader:
        box_loss, class_loss, ctr_loss = detector(inputs)
        print(box_loss.sum())
        print(class_loss.sum())
        print(ctr_loss.sum())
        optimizer.zero_grad()
        loss = box_loss.mean() + class_loss.mean() + ctr_loss.mean()
        print(loss)
        loss.backward()
        optimizer.step()
        epoch_loss.append(float(loss))
        total_loss = np.mean(epoch_loss)

    steps += 1

    scheduler.step(np.mean(epoch_loss))








