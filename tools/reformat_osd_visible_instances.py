import torch
import pickle

if __name__ == "__main__":
    
    pred_dict_list = []
    pred_anno_list = torch.load("/home/zhangjinyu/code_repository/uoais/output/R50_rgbdconcat_mlc_occatmask_hom_concat/inference/osd_visible_instances.pth")
    for instances in pred_anno_list:
        pred_ins = {
            "pred_vm": instances.pred_visible_masks.numpy(),
            "pred_fm": instances.pred_masks.numpy(),
        }
        pred_dict_list.append(pred_ins)
        
    with open("/home/zhangjinyu/code_repository/uoais/pred_dict_list.pkl", "wb") as f:
        pickle.dump(pred_dict_list, f)