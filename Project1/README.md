## <font face="楷体">说明</font>
基于背景建模的前景检测与基于Faster R-CNN的视频目标检测    

## 前景检测  
使用了帧差法、中值滤波法、高斯混合模型实现了视频前景检测  
### 帧差法、中值滤波

![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/zhencha1.png)
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/zhencha2.png)
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/zhencha3.png)

![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/zhongzhi1.png)
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/zhongzhi2.png)
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/zhongzhi3.png)

### 高斯混合模型

**opencv的版本需要为2或3，opencv 4版本会在image, cnts, hierarchy = cv2.findContours()处报错**

```
python GMM.py
```
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/GMM1.png)
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/GMM2.png)
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/GMM3.png)

## 目标检测  
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/FasterRCNN1.png)
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/FasterRCNN2.png)
![](https://github.com/Huntersxsx/SJTU_VideoAnalysis/tree/master/Project1/img/FasterRCNN3.png)
