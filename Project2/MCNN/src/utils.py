import cv2
import numpy as np
import os

def save_results(input_img, gt_data,density_map,output_dir, fname='results.png'):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[1],input_img.shape[0]))
        gt_data = cv2.resize(gt_data, (input_img.shape[1],input_img.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    cv2.imwrite(os.path.join(output_dir,fname),result_img)
    

def save_density_map(density_map,output_dir, fname='results.png'):    
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    cv2.imwrite(os.path.join(output_dir,fname),density_map)
    
def display_results(input_img, gt_data,density_map):
    print('1')
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
         input_img = cv2.resize(input_img, (density_map.shape[1],density_map.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    result_img  = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)

def save_counter(data_path, fname, et_count, save_dir):
    #input_img = input_img[0][0]
    #print(fname)
    cnt = int(et_count)
    #print(cnt)
    img = cv2.imread(os.path.join(data_path,fname))
    #cv2.imshow("ori_image", img)
    text = "There are "+ str(cnt) + " people in the pic"  
    #text = "画面中有"+ str(cnt) + "个人" 
    cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_PLAIN, 4.5, (0, 255, 0), 4)
    #cv2.imshow("ori_image", img)
    #key = cv2.waitKey(delay=1)
    cv2.imwrite(os.path.join(save_dir,fname), img)
    