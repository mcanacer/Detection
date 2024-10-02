import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class VOC2007DetectionTiny(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_dir,
            split='train',
            download=False,
            image_size=224,
            max_num_instances=40,
    ):
        super(VOC2007DetectionTiny, self).__init__()

        self.image_size = image_size
        self._max_num_instances = max_num_instances

        if download:
            self._attempt_download(dataset_dir)

        voc_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]

        self.class_to_idx = {
            _class: _idx for _idx, _class in enumerate(voc_classes)
        }
        self.idx_to_class = {
            _idx: _class for _idx, _class in enumerate(voc_classes)
        }

        self.instances = json.load(
            open(os.path.join(dataset_dir, f"voc07_{split}.json"))
        )
        self.dataset_dir = dataset_dir

        _transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
        self.image_transform = transforms.Compose(_transforms)

    @staticmethod
    def _attempt_download(dataset_dir: str):
        import wget

        os.makedirs(dataset_dir, exist_ok=True)

        wget.download(
            "https://web.eecs.umich.edu/~justincj/data/VOCtrainval_06-Nov-2007.tar",
            out=dataset_dir,
        )
        wget.download(
            "https://web.eecs.umich.edu/~justincj/data/voc07_train.json",
            out=dataset_dir,
        )
        wget.download(
            "https://web.eecs.umich.edu/~justincj/data/voc07_val.json",
            out=dataset_dir,
        )

        import tarfile

        voc_tar = tarfile.open(
            os.path.join(dataset_dir, "VOCtrainval_06-Nov-2007.tar")
        )
        voc_tar.extractall(dataset_dir)
        voc_tar.close()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        image_path, annotations = self.instances[index]
        image_path = image_path.replace("./here/", "")
        image_path = os.path.join(self.dataset_dir, image_path)
        image = Image.open(image_path).convert("RGB")

        gt_boxes = torch.tensor([ann["xyxy"] for ann in annotations])
        gt_labels = torch.tensor([self.class_to_idx[ann["name"]] + 1 for ann in annotations]).to(dtype=torch.long)
        gt_labels = gt_labels.unsqueeze(1)

        original_width, original_height = image.size

        normalize_tens = torch.tensor([original_width, original_height, original_width, original_height])

        gt_boxes /= normalize_tens[None, :]

        image = self.image_transform(image)

        if self.image_size is not None:
            if original_height >= original_width:
                new_width = self.image_size
                new_height = original_height * self.image_size / original_width
            else:
                new_height = self.image_size
                new_width = original_width * self.image_size / original_height

        _x1 = (new_width - self.image_size) // 2
        _y1 = (new_height - self.image_size) // 2

        gt_boxes[:, 0] = torch.clamp(gt_boxes[:, 0] * new_width - _x1, min=0)
        gt_boxes[:, 1] = torch.clamp(gt_boxes[:, 1] * new_height - _y1, min=0)
        gt_boxes[:, 2] = torch.clamp(gt_boxes[:, 2] * new_width - _x1, max=self.image_size)
        gt_boxes[:, 3] = torch.clamp(gt_boxes[:, 3] * new_height - _y1, max=self.image_size)

        gt_boxes /= self.image_size
        gt_weights = torch.ones_like(gt_labels)

        num_pad = self._max_num_instances - gt_boxes.shape[0]

        gt_boxes = F.pad(gt_boxes, (0, 0, 0, num_pad))
        gt_labels = F.pad(gt_labels, (0, 0, 0, num_pad), value=1).squeeze()
        gt_weights = F.pad(gt_weights, (0, 0, 0, num_pad)).squeeze()

        return image, gt_boxes, gt_labels, gt_weights
