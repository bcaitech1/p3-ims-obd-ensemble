# p3-ims-obd-ensemble

> BoostCamp AI Tech P Stage stage3 Image Segmentation
## List

* Segmentation   
      * JooYoung_dev   
      * HyeongMing_dev   
      * JunCheol_dev    
      * Kimin_dev    
      * MinJung_dev   
      * NuRee_dev   
* Detection   
      * JooYoung_dev   
      * HyeongMing_dev   
      * JunCheol_dev    
      * Kimin_dev    
      * MinJung_dev   
      * NuRee_dev   


## Segmentation    

### [JooYoung_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/JooYoung_dev/segmentations)

You can see this branch at main too.   

```
$> tree -d
.
├── segmentaions
|     ├── config
|     ├── data
|     ├── losses
|     ├── models
|     ├── scheduler
|     ├── utils
│     └── ...
└── detection
      └── ...
```

* **config**    
      * efficient b2/b6 + unet++    
      * efficient b4/b6 + deepLabV3+    
      * effunext    
      * resnext101 + upernet    
      * resnext50 + deepLabV3+     
      * unext    
* **losses**    
      * dice_ce_loss     
      * soft_ce_loss
* **scheduler**
      * customcosine

### [HyeongMin_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/HyeongMin_dev/Segmentation)

```
$> tree -d
.
├── Segmentaion
│     └── ...
└── detection
      └── ...
```

* **model**: resnext50_32x4d + DeepLabV3+ 

However, because used smp, you can use others in smp

### [JunCheol_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/JunCheol_dev/Segmentation)

```
$> tree -d
.
├── segmentaion
|     ├── ppt_paper
│     └── ...
└── detection
      └── ...
```

* **ppt_paper**   
      * segmentation_survey_2020.pdf     
      * Rethinking Pre-training and Self-training.pptx    
      * Imgage Segmentation.pptx    
      * EDA.pptx    
* **model**: emanet

### [Kimin_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/Kimin_dev/segmentations)

```
$> tree -d
.
├── segmentaions
│     └── ...
└── detection
      └── ...
```

* TTA_CRF
* psheudo_train

### [MinJung_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/MinJung_dev/seg)

```
$> tree -d
.
├── seg
|     ├── Tools
|     ├── dataset
|     ├── lib
|     ├── preprocess
│     └── ...
└── detection
      └── ...
```

* **lib**  
      * asgnet   
      * hrdnet   


### [NuRee_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/NuRee_dev/segmentation)

```
$> tree -d
.
├── segmentation
│     └── ...
└── detection
      └── ...
```

* augmentaion test   
* Resnext50 + DeepLabV3+  

## Object Detection    

### [JooYoung_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/JooYoung_dev/detection)

```
$> tree -d
.
├── segmentaions
│     └── ...
└── detection
      ├── UniverseNet-master
      └── ...
```  

**mmdet**

* universNet

### [HyeongMin_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/HyeongMin_dev/Object_Detection)

```
$> tree -d
.
├── segmentaion
│     └── ...
└── ObjectDetection
      ├── Swin-Transformer-Object-Detection
      └── ...
```

**mmdet**

* detectoRS + ResNeXt101


### [JunCheol_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/JunCheol_dev/ObjectDetection)

```
$> tree -d
.
├── segmentaion
│     └── ...
└── ObjectDetection
      ├── Swin-Transformer-Object-Detection
      └── ...
```

**mmdet**

* Swin-S   

### [Kimin_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/Kimin_dev/detection)

```
$> tree -d
.
├── segmentaions
│     └── ...
└── detection
      ├── efficientdet-naive-pytorch
      ├── faster_rcnn-naive-pytorch
      └── vfnetx-mmdet
```

**naive model**  

* efficientdet
* faster_rcnn

**mmdet**

* vfnetx

### [MinJung_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/MinJung_dev/detection)

```
$> tree -d
.
├── seg
│     └── ...
└── detection
      ├── Swin-Transformer-Object-Detection-master
      ├── mmdet
      └── augmentaion

```

**mmdet**
* Swin-T
* efficienet b3 - nas-fpn - cascade rcnn


### [NuRee_dev](https://github.com/bcaitech1/p3-ims-obd-ensemble/tree/NuRee_dev/detection)

```
$> tree -d
.
├── segmentation
│     └── ...
└── detection
      ├── Swin-Transformer-Object-Detection-master
      └── ...
```

**mmdet**

* Swin-T
