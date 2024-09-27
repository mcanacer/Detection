import torch
from dataset import VOC2007DetectionTiny
from resnet import *
from fpn import FPN
from retinanet import RetinaNet
import torch.optim as optim
import numpy as np
import wandb

wandb.login(key="ff9d5723d5542f318135cb8e45cf941976820f40")

run = wandb.init(
    name="early-submission",
    resume="must",
    id="fpu77upc",
    project="RetinaNet",
)

dataset_dir = "/Users/muhammetcan/Desktop/RetinaNet"

train_dataset = VOC2007DetectionTiny(dataset_dir, "train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, pin_memory=True, shuffle=True)

backbone = ResNet18()
FPN = FPN(backbone.out_channels)

if torch.cuda.is_available():
    backbone = backbone.cuda()
    FPN = FPN.cuda()

detector = RetinaNet(backbone, FPN, feature_size=256, num_anchors=9, num_classes=20)

if torch.cuda.is_available():
    detector = detector.cuda()

detector.load_state_dict(torch.load("/content/drive/MyDrive/RetinaNet/checkpoint.pth", weights_only=True)['model_state_dict'])

optimizer = optim.SGD(detector.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

num_epochs = 500
best_loss = 1e+5

for epoch in range(num_epochs):
    detector.train()

    curr_lr = float(optimizer.param_groups[0]['lr'])

    epoch_loss = []

    for idx, inputs in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs[0], inputs[1] = inputs[0].cuda(), inputs[1].cuda()
            cls_loss, reg_loss = detector(inputs, training=True)
        else:
            cls_loss, reg_loss = detector(inputs, training=True)
        loss = cls_loss + reg_loss
        torch.nn.utils.clip_grad_norm_(detector.parameters(), 0.1)
        loss.backward()
        optimizer.step()
        epoch_loss.append(float(loss))
        total_loss = np.mean(epoch_loss)

        wandb.log({
        "cls_loss": cls_loss,
        "reg_loss": reg_loss,
        "train_loss": loss,
        "epoch": epoch,
        "learning-rate": curr_lr})

    torch.save({'model_state_dict':detector.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'scheduler_state_dict':scheduler.state_dict(),
                     'epoch': epoch}, "./checkpoint.pth"
                     )

    print(np.mean(epoch_loss))

    scheduler.step(np.mean(epoch_loss))
