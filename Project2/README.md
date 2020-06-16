## <font face="楷体">说明</font>
使用了YOLO v3、MCNN、LSC-CNN、12-in-1，4种模型实现了高密度人群及低密度人群视频人群计数  

## YOLO v3
YOLO v3是一种One-Stage的目标检测算法，这类检测算法不需要Region Proposal阶段，可以通过一个Stage直接产生物体的类别概率和位置坐标值。YOLO v3使用了darknet-53作为特征提取的backbone，在精度上与Resnet相当，在计算速度上却得到了很大的提升。为了加强算法对小目标检测的精确度，YOLO v3中采用了类似FPN的upsample和融合做法，在多个scale的feature map上做检测。此外，在loss function种，作者替换了原有的用softmax获取类别得分并用最大得分的标签来表示包含再边界框内的目标，而对图像中检测到的对象执行多标签分类，就是对每种类别使用二分类的logistic回归。从效果来看，YOLO v3可以取得和SSD同样的精度，速度却提升了3倍，是精度与速度兼顾的模型。

使用了[leviome](https://github.com/leviome/human_counter)的代码，[paper](https://arxiv.org/abs/1804.02767)。  

首先下载预训练好的model，并将模型转为Keras model，参考[link](https://github.com/qqwweee/keras-yolo3#quick-start)。

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```   
使用YOLO v3对视频进行人群计数，运行flow.py文件 python flow.py [需要计数的视频路径] [计数后的视频输出路径]
```
## 示例：
python flow.py F:/SJTU_VideoAnalysis/input_video/demo.avi F:/SJTU_VideoAnalysis/YOLOv3/output/demo_output.avi
```

YOLO v3实现效果 (某一帧)，完整视频见PPT

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project2/img/YOLO1.png)

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project2/img/YOLO2.png)

## MCNN
MCNN是一种简单有效的多列卷积神经网络结构，可以将图像映射到对应的人群密度图上，允许输入任意尺寸或分辨率的图像，每列CNN学习得到的特征可以自适应由于透视或图像分辨率引起的人头大小的变化，并能在不需要输入图的透视先验情况下，通过几何自适应的核来精确计算人群密度图。  

使用了[svishwa](https://github.com/svishwa/crowdcount-mcnn)的代码，[paper](http://cinslab.com/wp-content/uploads/2019/03/2019_3_7_1.pdf)。  

根据[Data Setup](https://github.com/svishwa/crowdcount-mcnn#data-setup)准备好数据。  
使用test.py代码(已注释掉不必要部分，并将统计人数显示到图片上)。  

```
## 先将视频拆帧，input_video：需要拆帧的原视频，frame_dir：拆帧后帧图像保存到路径，v2i：视频to图像
## 示例：
python SplitCombine.py --frame_dir F:/SJTU_VideoAnalysis/MCNN/frames --input_video F:/SJTU_VideoAnalysis/input_video/demo.avi --v2i True

## 使用MCNN对拆帧后的图像进行人群计数
## test.py文件中，data_path =  'F:/SJTU_VideoAnalysis/MCNN/frames/'是需要进行人群计数的图像路径，处理后的图像保存在./output/results/中
## 示例：
cd MCNN
python test.py

## 将处理后的图像合并成视频，frame_dir：需要合并的图像路径，output_video：合并后视频的输出路径，i2v：图像to视频
## 示例：
cd ..
python SplitCombine.py --frame_dir F:/SJTU_VideoAnalysis/MCNN/output/result --output_video F:/SJTU_VideoAnalysis/MCNN/output/demo_output.avi --i2v True
```

MCNN实现效果 (某一帧)，完整视频见PPT

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project2/img/MCNN1.png)

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project2/img/MCNN2.png)

## LSC-CNN
LSC-CNN是一个利用检测的方法进行人群计数的模型，它设计了一个新颖的CNN框架，可以在高分辨率图像上精确定位人头，此外，作者还设计了一个与从上到下反馈结构相融合的方案，使得网络可以联合处理多尺度信息，方便网络更好地定位人头。在仅有点标注信息的情况下，可以预测每个人头的bounding box，并且在GWTA模块使用了新设计的winner-take-all的loss，有利于在高分辨率的图像上进行训练。  

使用了[val-iisc](https://github.com/val-iisc/lsc-cnn)和[vlad3996](https://github.com/vlad3996/lsc-cnn)的代码，[paper](https://arxiv.org/abs/1906.07538)。 

下载[权重](https://drive.google.com/drive/folders/1A_619zNVWvKsI9w7lS6OfP536TXMWLjf)，存放在weights文件夹中。  

```
## 先将视频拆帧，input_video：需要拆帧的原视频，frame_dir：拆帧后帧图像保存到路径，v2i：视频to图像
## 示例：
python SplitCombine.py --frame_dir F:/SJTU_VideoAnalysis/LSC-CNN/frames --input_video F:/SJTU_VideoAnalysis/input_video/demo.avi --v2i True

## 使用MCNN对拆帧后的图像进行人群计数
## predict.py文件中，data_path =  'F:/SJTU_VideoAnalysis/LSC-CNN/frames/'是需要进行人群计数的图像路径，处理后的图像保存在./output/results/中
## 示例：
cd LSC-CNN
python predict.py

## 将处理后的图像合并成视频，frame_dir：需要合并的图像路径，output_video：合并后视频的输出路径，i2v：图像to视频
## 示例：
cd ..
python SplitCombine.py --frame_dir F:/SJTU_VideoAnalysis/LSC-CNN/output/result --output_video F:/SJTU_VideoAnalysis/LSC-CNN/output/demo_output.avi --i2v True
``` 

LSC-CNN实现效果 (某一帧)，完整视频见PPT

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project2/img/LSC-CNN1.png)

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project2/img/LSC-CNN2.png)

## 12-in-1
12-in-1模型是通过多任务训练来学习视觉语言联合表示的一种跨模态模型，该模型涉及了四类任务，视觉问题回答 (Visual Question Answering)，基于图像描述的图像检索 (Caption-based Image Retrieval)，看图识物 (Grounding Referring Expressions) 和多模态验证 (Multi-modal Verification)，并在12 个不同的数据集上进行联合训练。  
通过多任务的学习可以获得更广泛的视觉语言联合表示，并用于不同的下游任务中，在本次人群计数的实验中，将12-in-1模型应用到VQA这个下游任务中，输入视频的帧图像，以及问题“How many people are there in the picture?”，从而获得画面中人物的数量。  

使用了作者提供的网页版[Demo](https://vilbert.cloudcv.org/)，[code](https://github.com/facebookresearch/vilbert-multi-task), [paper](https://arxiv.org/abs/1912.02315)。  

12-in-1实现效果

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project2/img/12-in-1-1.png)

![](https://github.com/Huntersxsx/SJTU-VideoAnalysis/blob/master/Project2/img/12-in-1-2.png)

## 其他
- 这个链接整理了常见的人群计数模型的code与paper，[link](https://github.com/gjy3035/Awesome-Crowd-Counting)  
- OpenCV实现人群计数，[link](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/)