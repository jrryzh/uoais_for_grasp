import torch
import json
import glob
import cv2
import os
import pickle
import cvbase as cvb
import itertools
import numpy as np
from pycocotools import mask as maskUtils

def compute_iou_rle(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    iou = intersection / union if union != 0 else 0
    return iou

if __name__ == "__main__":
    
    best_vm_matches = []
    
    dataset_path = "/home/zhangjinyu/code_repository/uoais/datasets/OSD-0.2-depth"
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(dataset_path)))
    amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(dataset_path)))
    occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(dataset_path)))

    pred_anno_list = torch.load("/home/zhangjinyu/code_repository/uoais/output/R50_rgbdconcat_mlc_occatmask_hom_concat/inference/osd_visible_instances.pth")
    
    for j, gt_anno_path in enumerate(amodal_anno_paths):
        max_iou = 0 
        max_i = None
        max_pred_vm = None
        
        anno_file = amodal_anno_paths[j]
        amodal_anno = cv2.imread(anno_file)[...,0]
        img_name = anno_file.split('/')[-1].split('_')[0] + '.png'
        anno_id = int(anno_file.split('/')[-1].split('_')[1].strip('.png'))
        gt_vm = cv2.imread(os.path.join("/home/zhangjinyu/code_repository/uoais/datasets/OSD-0.2-depth/annotation", img_name))[...,0] == anno_id

    
        full_name = "/home/zhangjinyu/code_repository/uoais/datasets/OSD-0.2-depth/image_color/" + img_name
        index = rgb_paths.index(full_name)
        
        pred_instance = pred_anno_list[index]
        visibles = pred_instance.pred_visible_masks.detach().cpu().numpy() 
        
        for i in range(visibles.shape[0]):
            pred_vm = visibles[i]
            iou = compute_iou_rle(gt_vm, pred_vm)
            if iou > max_iou:
                max_iou = iou
                max_i = i
                max_pred_vm = pred_vm
                
        best_vm_matches.append(max_pred_vm)
        print(f"finished processing image {j}")
    
    filename = '/home/zhangjinyu/code_repository/uoais/output/best_vm_matches_uoais.pkl'
    # 假设 best_vm_matches 是你想要保存的对象
    with open(filename, 'wb') as f:
        pickle.dump(best_vm_matches, f)
            
            
            
            
        
        