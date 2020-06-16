import cv2
import matplotlib.pyplot as plt
from model import LSCCNN
import os
import numpy as np

checkpoint_path = './weights/part_b_scale_4_epoch_24_weights.pth'
save_dir = "./output/"
data_path = "F:/PETS2006/newframes/"

#output_dir = './output/'
#model_name = os.path.basename(model_path).split('.')[0]
#file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, 'results')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

network = LSCCNN(checkpoint_path=checkpoint_path)
#network.cuda()
network.eval();

def save_results(data_path, fname, save_dir):
    img = cv2.imread(os.path.join(data_path,fname))
    print('Loaded ', fname)
    #image = cv2.imread('F:/PETS2006/manypeople/IMG_158.jpeg')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred_dot_map, pred_box_map, img_out = network.predict_single_image(img, nms_thresh=0.75)
    cnt = np.sum(pred_dot_map)

    text = "There are " + str(cnt) + " people in the pic"  
    print(text)
    cv2.putText(img_out, text, (50, 100), cv2.FONT_HERSHEY_PLAIN, 4.5, (0, 255, 0), 4)
    cv2.imwrite(os.path.join(save_dir,fname), img_out)

if __name__ == '__main__':
    data_files = [filename for filename in sorted(os.listdir(data_path)) \
                           if os.path.isfile(os.path.join(data_path,filename))]
    data_files.sort(key=lambda x: int(x.split('.')[0][4:]))

    for fname in data_files:
        save_results(data_path, fname, save_dir)



#plt.figure(figsize=(18,10))
#plt.imshow(img_out)
#plt.show()