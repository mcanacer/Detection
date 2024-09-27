import torch
import torch.nn as nn
from losses import Focal, BoxLoss
from assigners import ArgmaxAssigner
import math
from torchvision.ops.boxes import nms as nms_torch
from utils import Anchors, BBoxTransform, ClipBoxes
from similarity_calculators import IouSimilarity


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class Regressor(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        output = inputs.permute(0, 2, 3, 1)
        return output.contiguous().view(output.shape[0], -1, 4)


class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        inputs = self.act(inputs)
        inputs = inputs.permute(0, 2, 3, 1)
        output = inputs.contiguous().view(inputs.shape[0], inputs.shape[1], inputs.shape[2], self.num_anchors,
                                          self.num_classes)
        return output.contiguous().view(output.shape[0], -1, self.num_classes)


class RetinaNet(nn.Module):
    def __init__(self, backbone, fpn, feature_size, num_anchors, num_classes):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.feature_size = feature_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.classifier = Classifier(fpn.feature_size, num_anchors, num_classes, 4)
        self.assigner = ArgmaxAssigner(IouSimilarity(), 0.5, 0.4)
        self.regressor = Regressor(fpn.feature_size, num_anchors, 4)
        self.focalLoss = Focal()
        self.regLoss = BoxLoss()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.anchors = Anchors()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)

    def forward(self, inputs, training=None):
        images, targets = inputs
        if training:
            gt_boxes, gt_classes = targets[:, :, :4], targets[:, :, 4]

        features = self.backbone(images)
        features = self.fpn(features)

        cls_pred = torch.cat([self.classifier(feature) for feature in features], dim=1)
        loc_pred = torch.cat([self.regressor(feature) for feature in features], dim=1)

        anchors = self.anchors(images)

        matched_idx = self.assigner(gt_boxes, anchors)

        if training:
            cls_loss = self.focalLoss(cls_pred, gt_classes, matched_idx)
            box_loss = self.regLoss(loc_pred, gt_boxes, anchors, matched_idx)
            return cls_loss, box_loss
        else:
            transformed_anchors = self.regressBoxes(anchors, loc_pred)
            transformed_anchors = self.clipBoxes(transformed_anchors, images)

            scores = torch.max(cls_pred, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = cls_pred[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


