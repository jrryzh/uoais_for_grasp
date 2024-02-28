import torch
import json
import cvbase as cvb
import itertools
from pycocotools import mask as maskUtils

def compute_iou_rle(mask1, mask2):
    """
    计算两个RLE编码mask之间的IoU。
    """
    i = maskUtils.area(maskUtils.merge([mask1, mask2], intersect=True))
    u = maskUtils.area(maskUtils.merge([mask1, mask2], intersect=False))
    return i / float(u) if u > 0 else 0

# list(itertools.chain(*[x["instances"] for x in self._amodal_predictions]))
if __name__ == "__main__":
    
    best_vm_matches = []
    
    gt_anno_list = cvb.load("/home/zhangjinyu/code_repository/uoais/datasets/UOAIS-Sim/annotations/coco_anns_uoais_sim_val.json")["annotations"]
    pred_anno_list = torch.load("/home/zhangjinyu/code_repository/uoais/output/R50_rgbdconcat_mlc_occatmask_hom_concat/inference/instances_visible_predictions.pth")
    
    for j, gt_anno in enumerate(gt_anno_list):
        max_iou = 0 
        max_i = None
        max_pred_vm = None
            
        gt_image_id = gt_anno["image_id"]
        gt_vm = gt_anno["visible_mask"]
        
        pred_instances_image_id = pred_anno_list[gt_image_id]["image_id"]
        assert gt_image_id == pred_instances_image_id
        pred_instances_lst = pred_anno_list[gt_image_id]["instances"]
        
        # {'image_id': 0, 'category_id': 0, 'bbox': [0, 0, 0, 0], 'score': 1.0, 'segmentation': {'size': [480, 640], 'counts': 'PP\\9'}, 'area': 0.0}
        for i, pred_instance in enumerate(pred_instances_lst):
            pred_vm = pred_instance["segmentation"]
            iou = compute_iou_rle(gt_vm, pred_vm)
            if iou > max_iou:
                max_iou = iou
                max_i = i
                max_pred_vm = pred_vm
                
        best_vm_matches.append(max_pred_vm)
        print(f"finished processing image {j}")
    
    filename = '/home/zhangjinyu/code_repository/uoais/output/best_vm_matches.json'
    # 使用with语句打开文件以保证正确的资源管理
    with open(filename, 'w') as file:
        # 使用json.dump将数据写入文件，确保指定`ensure_ascii=False`以支持非ASCII字符
        # json.dump(best_vm_matches, file, ensure_ascii=False, indent=4)
        json.dump(best_vm_matches, file)
        
    print(f"数据已保存到文件：{filename}")
            
            
            
            
        
        