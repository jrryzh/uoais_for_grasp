import argparse
import numpy as np
import os

from eval.eval_utils import eval_visible_on_OSD, eval_amodal_occ_on_OSD

if __name__ == "__main__":

    parser = argparse.ArgumentParser('UOIS CenterMask', add_help=False)

    # model config   
    parser.add_argument("--config-file", 
        default="./configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml", 
        metavar="FILE", help="path to config file")    
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--vis-only", action="store_true")
    parser.add_argument(
        "--use-cgnet",
        action="store_true",
        help="Use foreground segmentation model to filter our background instances or not"
    )
    parser.add_argument(
        "--cgnet-weight-path",
        type=str,
        default="./foreground_segmentation/rgbd_fg.pth",
        help="path to forground segmentation weight"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/OSD-0.2-depth",
        help="path to the OSD dataset"
    )


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    rgb_path = None
    depth_path = None

    # load rgb and depth
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.resize(rgb_img, (W, H))
    depth_img = imageio.imread(depth_path)
    depth_img = normalize_depth(depth_img)
    depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_img = inpaint_depth(depth_img)
        
    # UOAIS-Net inference
    if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
        uoais_input = depth_img
    elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
        uoais_input = np.concatenate([rgb_img, depth_img], -1)   
    else:
        uoais_input = rgb_img
        
    # load GT (amodal masks)
    img_name = os.path.basename(rgb_path)[:-4]
    annos = [] # [instance, IMG_H, IMG_W]
    filtered_amodal_paths = list(filter(lambda p: img_name + "_" in p, amodal_anno_paths))
    filtered_occlusion_paths = list(filter(lambda p: img_name + "_" in p, occlusion_anno_paths))

    for anno_path in filtered_amodal_paths:
        # get instance id  
        inst_id = os.path.basename(anno_path)[:-4].split("_")[-1]
        inst_id = int(inst_id)
        # load mask image
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        # fill mask with instance id
        cnd = anno > 0
        anno_mask = np.zeros((H, W))
        anno_mask[cnd] = inst_id
        annos.append(anno_mask)            
    annos = np.stack(annos)
    num_inst_all_gt += len(filtered_amodal_paths)

    # forward (UOAIS)
    outputs = predictor(uoais_input)
    instances = detector_postprocess(outputs['instances'], H, W).to('cpu')
        
    # with open("/home/zhangjinyu/code_repository/uoais/output/R50_rgbdconcat_mlc_occatmask_hom_concat/inference/osd_visible_instances.pth", "wb") as f:
    #     torch.save(vm_instances_lst, f)

    if not args.use_cgnet:
        pred_masks = instances.pred_masks.detach().cpu().numpy()
        preds = [] # mask per each instance
        for i, mask in enumerate(pred_masks):
            pred = np.zeros((H, W))
            pred[mask > False] = i+1
            preds.append(pred)
            num_inst_all_pred += 1
    else:
        fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
        fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
        fg_depth_input = cv2.resize(depth_img, (320, 240)) 
        fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
        fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
        fg_output = fg_model(fg_input.cuda())
        fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
        fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
        fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

        # filter amodal predictions with foreground mask
        pred_masks = instances.pred_masks.detach().cpu().numpy()
        preds = [] # mask per each instance
        for i, mask in enumerate(pred_masks):
            overlap = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
            if overlap >= 0.5: # filiter outliers
                pred = np.zeros((H, W))
                pred[mask > False] = i+1
                preds.append(pred)
                num_inst_all_pred += 1

    if len(preds) > 0:
        preds = np.stack(preds)
    else:
        preds = np.array(preds)

    amodals = instances.pred_masks.detach().cpu().numpy()
    visibles = instances.pred_visible_masks.detach().cpu().numpy()        
        