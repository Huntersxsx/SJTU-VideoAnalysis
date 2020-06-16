# LSC-CNN

### Simple test on image

```python
import cv2
from model import LSCCNN

checkpoint_path = './weights/part_b_scale_4_epoch_24_weights.pth'

network = LSCCNN(checkpoint_path=checkpoint_path)
network.cuda()
network.eval();

image = cv2.imread('./dataset/ST_partB/test_data/images/IMG_2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pred_dot_map, pred_box_map, img_out = network.predict_single_image(image, nms_thresh=0.25)

plt.figure(figsize=(18,10))
plt.imshow(img_out)
plt.show()

```

<img src="https://github.com/vlad3996/lsc-cnn/blob/3d8887b603aec5d68adad47c51d085fe4c6de50b/download.png"/>


This repository is a fork of the pytorch implementation for the crowd counting model, LSC-CNN, proposed in the paper - [**Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection**](https://arxiv.org/pdf/1906.07538.pdf).

```
@article{LSCCNN19,
    Author = {Sam, Deepak Babu and Peri, Skand Vishwanath and Mukuntha .N .S ,  and Kamath, Amogh and Babu, R. Venkatesh},
    Title = {Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection},
    Journal = {arXiv:1906.07538},
    Year = {2019}
}
```

