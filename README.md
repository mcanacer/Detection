Retinanet From Scratch (https://arxiv.org/abs/1708.02002)

My model has 6.3 million parameters. I used mobilenet v3 as backbone. Fpn levels are from 3 to 7. Convolution towers and convolution heads have num repeats 1. I used L1 loss for regression. Focal loss for classification. Finally, I used iou loss

I used wandb to see the graphs while training. The graphs are in this link. https://wandb.ai/muhammedacer-mca-Istanbul%20University-Cerrahpa%C5%9Fa/RetinaNet/runs/am7h6wbf?nw=nwusermuhammedacermca

Performance Table 

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.296  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.184  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.013  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.274  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.174  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.215  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.215  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.013  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.148  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.332  
