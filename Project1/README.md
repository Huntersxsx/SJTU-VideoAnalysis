## <font face="楷体">说明</font>
基于背景建模的前景检测与基于Faster R-CNN的视频目标检测    

## 前景检测  
使用了帧差法、中值滤波法、高斯混合模型实现了视频前景检测  
### 帧差法、中值滤波

```
python ZhenCha.py
```

帧差法实现效果 (某一帧)，完整视频见PPT
![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project1/img/zhencha3.png)

中值滤波实现效果 (某一帧)，完整视频见PPT
![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project1/img/zhongzhi3.png)

### 高斯混合模型

**opencv的版本需要为2或3，opencv 4版本会在image, cnts, hierarchy = cv2.findContours()处报错**

```
python GMM.py
```

GMM实现效果 (某一帧)，完整视频见PPT
![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project1/img/GMM3.png)

## 目标检测  

使用了[jwyang的代码](https://github.com/jwyang/faster-rcnn.pytorch)  
这里仅使用了代码中的[Demo](https://github.com/jwyang/faster-rcnn.pytorch#demo)部分,用训练好的模型来测试自己的图片 
首先，根据[compilation](https://github.com/jwyang/faster-rcnn.pytorch#compilation)配置好环境  

```
git clone https://github.com/jwyang/faster-rcnn.pytorch.git
pip install -r requirements.txt
cd lib
sh make.sh
```

然后下载预训练好的模型，这里采用的模型faster_rcnn_1_7_10021.pth是在PASCAL VOC 2007数据集上训练的，backbone为ResNet101。[link](https://www.dropbox.com/s/4v3or0054kzl19q/faster_rcnn_1_7_10021.pth?dl=0)  
将下载好的模型faster_rcnn_1_7_10021.pth放在$ROOT/models/res101/pascal_voc/faster_rcnn_1_7_10021.pth
其余预训练模型见[link](https://github.com/jwyang/faster-rcnn.pytorch#benchmarking)

由于Faster R-CNN是针对图片的目标检测，所以需要将视频进行拆帧，对每一张帧图像进行目标检测，然后再把检测后的所有帧图像进行合并成视频  
这里用到了代码SplitCombine.py，主要借助了OpenCV工具

```
## 先将视频拆帧，input_video：需要拆帧的原视频，frame_dir：拆帧后帧图像保存到路径，v2i：视频to图像
## 示例：
python SplitCombine.py --frame_dir F:/SJTU_VideoAnalysis/frames --input_video F:/SJTU_VideoAnalysis/input_video/demo.avi --v2i True

## 使用Faster R-CNN对拆帧后的图像进行目标检测，因为使用的预训练模型为faster_rcnn_1_7_10021.pth，
所以--net resnet101  --checksession 1  --checkepoch 7 --checkpoint 10021
image_dir: 需要被目标检测的图像路径，save_dir：目标检测后图像的保存路径
## 示例：
python demo.py --net resnet101  --checksession 1  --checkepoch 7 --checkpoint 10021 --load_dir models --image_dir F:/SJTU_VideoAnalysis/frames --save_dir F:/SJTU_VideoAnalysis/output_frames

## 将检测后的图像合并成视频，frame_dir：需要合并的图像路径，output_video：合并后视频的输出路径，i2v：图像to视频
## 示例：
python SplitCombine.py --frame_dir F:/SJTU_VideoAnalysis/output_frames --output_video F:/SJTU_VideoAnalysis/output_video/demo_output.avi --i2v True
```

Faster R-CNN实现效果 (某一帧)，完整视频见PPT

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project1/img/FasterRCNN3.png)

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project1/img/FasterRCNN4.jpg)