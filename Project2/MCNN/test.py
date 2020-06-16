import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
import cv2


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path =  'F:/SJTU_VideoAnalysis/MCNN/frames/'
gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'
model_path = './final_models/mcnn_shtechA_660.h5'
save_dir = "./output/"

#output_dir = './output/'
#model_name = os.path.basename(model_path).split('.')[0]
#file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, 'results')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)



net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
#net.cuda()
net.eval()
#mae = 0.0
#mse = 0.0

#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=True)


for blob in data_loader:                        
    im_data = blob['data']
    #gt_data = blob['gt_density']
    gt_data = None
    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    #gt_count = np.sum(gt_data)
    gt_count = 0
    et_count = np.sum(density_map)
    print(et_count)
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        #utils.save_counter(data_path, blob['fname'], et_count, save_dir)
        utils.save_density_map(density_map, save_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        #utils.save_results(im_data, gt_data, density_map, save_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        
#mae = mae/data_loader.get_num_samples()
#mse = np.sqrt(mse/data_loader.get_num_samples())
#print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

#f = open(file_results, 'w') 
#f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
#f.close()