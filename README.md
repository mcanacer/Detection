Retinanet From Scratch (https://arxiv.org/abs/1708.02002)

Oncelikle, modelimi google colabda egittim. Egitim icin A100 GPU sunu kullandim. Modelimde 6,3 milyon parametre var. Backbone olarak mobilenet v3 kullandım. Fpn seviyeleri 3 ile 7 arasındadır. Convolution tower ve convolution head 1 tekrar sayısına sahiptir. Regression için L1 loss kullandım. Class için focal loss. Son olarak, boxlar icin iou loss kullandım. Modeli argmax assigner ile egittim.   

Eğitim sırasında grafikleri görmek için wandb kutuphanesini kullandım. Grafikler bu linkte. https://wandb.ai/muhammedacer-mca-Istanbul%20University-Cerrahpa%C5%9Fa/RetinaNet/runs/am7h6wbf?nw=nwusermuhammedacermca

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

------------------------------------------------------------------------------

Adaptive Training Sample Selection (https://arxiv.org/abs/1912.02424)
Bu makaledeki assigner implementasyonunu (anchor-based Retinanet) yaptim. 

------------------------------------------------------------------------------

FCOS From Scracth (https://arxiv.org/abs/1904.01355)
Modelimi yine google colabda egittim. Egitim icin A100 GPU sunu kullandim. Modelimde 5,6 milyon parametre var. Backbone olarak mobilenet v3 kullandım. Fpn seviyeleri 3 ile 7 arasındadır. Convolution tower ve convolution head 1 tekrar sayısına sahiptir. Regression için L1 loss kullandım. Class için focal loss. Son olarak, centerness loss icin binary cross entropy loss u kullandim makalede yazildigi gibi.


Eğitim sırasında grafikleri görmek için wandb kutuphanesini kullandım. Grafikler bu linkte. https://wandb.ai/muhammedacer-mca-Istanbul%20University-Cerrahpa%C5%9Fa/FCOS/runs/nbllu0f4?nw=nwusermuhammedacermca

Performans Tablosu 

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.064  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.139  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.051  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.019  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.108  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.097  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.124  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.044  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.204  
