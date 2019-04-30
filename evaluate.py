import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


def preprocessing(image, CONFIG):
    # Resize
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    # image = image.to(device)

    return image, raw_image


def inference_mask(model, image, bk, raw_image=None, postprocessor=None):
    # Image -> Probability map
    logits, cascade_masks = model(image, bk)

    pred_mask = cascade_masks[-1]
    
    probs = torch.cat((1.0 - pred_mask, pred_mask), dim=1)[0]
    probs = probs.cpu().numpy()

    _, H, W = probs.shape

    raw_image = cv2.resize(raw_image, (W, H))
    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs_prev = probs
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)
    alpha = probs[1, :, :]

    return labelmap, alpha


class DomeSegmentEvalDataset(Dataset):
    def __init__(self, mean_bgr, CONFIG, all_gray=False):
        super(DomeSegmentEvalDataset).__init__()
        
        self.CONFIG = CONFIG
        self.all_gray = all_gray
        self.mean_bgr = mean_bgr

        val_root = '/media/mscv1/14fecf86-bdfa-4ebd-8b47-eea4ddee198e/dome_val_imgs'
        fg_folder = 'fgs'
        bg_folder = 'bgs'
        gt_folder = 'gts'
        fg_folder_path = osp.join(val_root, fg_folder)
        bg_folder_path = osp.join(val_root, bg_folder)
        gt_folder_path = osp.join(val_root, gt_folder)

        self.val_root = val_root
        self.fg_folder_path = fg_folder_path
        self.bg_folder_path = bg_folder_path
        self.gt_folder_path = gt_folder_path

        self.sample_list = self.get_sample_list()


    def img_transform(self, image):
        image = image.astype(np.float32)
        # Mean subtraction
        image -= self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return image


    def get_sample_list(self):
        img_paths = list()
        bk_paths = list()
        sample_ids = list()
        gt_paths = list()

        for root, dirs, files in os.walk(self.fg_folder_path):
            for name in files:
                img_path = osp.join(root, name)

                bk_path = osp.join(self.bg_folder_path, name)
                sample_sub_path = img_path[len(self.fg_folder_path)+1:]
                sample_id = sample_sub_path.replace('/', '_').split('.')[0]

                gt_path = osp.join(self.gt_folder_path, sample_sub_path)

                img_paths.append(img_path)
                bk_paths.append(bk_path)
                sample_ids.append(sample_id)
                gt_paths.append(gt_path)

        return list(zip(img_paths, bk_paths, sample_ids, gt_paths))

    
    def __getitem__(self, index):
        img_path, bk_path, sample_id, gt_path = self.sample_list[index]
        if not self.all_gray:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            bk = cv2.imread(bk_path, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            bk = cv2.imread(bk_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            bk = cv2.cvtColor(bk, cv2.COLOR_GRAY2BGR)
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # img_tensor = self.img_transform(img)
        # bk_tensor = self.img_transform(bk)
        img_tensor, _ = preprocessing(img, self.CONFIG)
        bk_tensor, _ = preprocessing(bk, self.CONFIG)

        return img_tensor[0], bk_tensor[0], img, bk, gt, sample_id


    def __len__(self):
        return len(self.sample_list)


def filter_segment_hilo3(alpha, thresh_hi=0.9, thresh_lo=0.3):
    mask_lo = np.where(alpha > thresh_lo, 255, 0).astype(np.uint8)
    
    output = cv2.connectedComponentsWithStats(mask_lo, connectivity=4, ltype=cv2.CV_16U)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    id_area_pair = [(stats[idx, cv2.CC_STAT_AREA], idx) for idx in range(1, num_labels)]
    remain_id_area_pairs = [pair for pair in id_area_pair if pair[0] > 10000]
    #print(id_area_pair)
    #print(remain_id_area_pairs)

    new_mask = np.zeros(mask_lo.shape, dtype=np.uint8)
    for comp_area, comp_id in remain_id_area_pairs:
        comp = labels == comp_id
        comp_hi_area = np.sum(comp * (alpha > thresh_hi))
        #print(comp_hi_area)
        if float(comp_hi_area) / comp_area > thresh_hi:
            new_mask[comp] = 255

    return new_mask


def eval_mask(mask_pred, mask_gt):
    pos_mask = (mask_pred > 0)
    neg_mask = (mask_pred <= 0)
    
    gt_mask = (mask_gt > 0)

    tp_mask = (pos_mask) * gt_mask
    fp_mask = (pos_mask) * (1 - gt_mask)
    fn_mask = (neg_mask) * gt_mask

    tp_area = np.sum(tp_mask)
    fp_area = np.sum(fp_mask)
    fn_area = np.sum(fn_mask)
    gt_area = np.sum(gt_mask)

    precision = float(tp_area) / (tp_area + fp_area)
    recall = float(tp_area / gt_area)
    
    # draw error map
    H, W = mask_pred.shape
    err_map = np.where(gt_mask, 255, 0).astype(np.uint8).reshape(H, W, 1)
    err_map = np.repeat(err_map, 3, axis=2)
    
    fp_pos = fp_mask > 0
    err_map[:, :, 0][fp_pos] = 255 # blue for fp
    err_map[:, :, 1][fp_pos] = 0 # blue for fp
    err_map[:, :, 2][fp_pos] = 0 # blue for fp
    
    fn_neg = fn_mask > 0
    err_map[:, :, 2][fn_neg] = 255 # red for fn
    err_map[:, :, 0][fn_neg] = 0 # red for fn
    err_map[:, :, 1][fn_neg] = 0 # red for fn

    return precision, recall, fp_area, fn_area, tp_area, err_map


class DomeSegmentationEval(object):
    def __init__(self, CONFIG, all_gray=False):
        self.CONFIG = CONFIG
        self.mean_bgr = np.array((CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R))
        self.val_dataset = DomeSegmentEvalDataset(self.mean_bgr, CONFIG, all_gray)
        device_cnt = torch.cuda.device_count()
        print('Evaluator using {} devices'.format(device_cnt))
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=device_cnt,
            num_workers=device_cnt,
            shuffle=False
        )


    def eval(self, model):
        # model should be DataParallel
        print('begin evaluation')
        model.eval()
        torch.set_grad_enabled(False)
        device = torch.device('cuda')
        loader_iter = iter(self.val_loader)
        
        eval_items = list()

        for img_tensors, bk_tensors, imgs, bks, gts, sample_ids in loader_iter:
            imgs = imgs.cpu().numpy()
            bks = bks.cpu().numpy()
            gts = gts.cpu().numpy()

            print('current batch: ' + str(sample_ids))
            img_tensors = img_tensors.to(device)
            bk_tensors = bk_tensors.to(device)
            
            logits, cascade_masks = model(img_tensors, bk_tensors)
            pred_masks = cascade_masks[-1]
            B, _, predH, predW = pred_masks.shape
            _, oriH, oriW, _ = imgs.shape

            probs = torch.cat((1.0 - pred_masks, pred_masks), dim=1)
            for bid in range(B):
                prob = probs[bid].detach().cpu().numpy()
                
                # may refine prob with post processor

                alpha = prob[1, :, :]
                alpha = cv2.resize(alpha, (oriW, oriH))
                
                eval_items.append((alpha, gts[bid], sample_ids[bid], imgs[bid], bks[bid]))

        eval_result_items = list()
        for eval_item in eval_items:
            alpha, gt, sample_id, img, bk = eval_item
            mask = filter_segment_hilo3(alpha, thresh_hi=0.9, thresh_lo=0.3)
            precision, recall, fp_area, fn_area, tp_area, err_map = eval_mask(mask, gt)
            eval_result_items.append((precision, recall, fp_area, fn_area, tp_area, err_map, sample_id))

        #avg_precision = np.mean([item[0] for item in eval_result_items])
        #avg_recall = np.mean([item[1] for item in eval_result_items])
        total_fp = np.sum([item[2] for item in eval_result_items])
        total_fn = np.sum([item[3] for item in eval_result_items])
        total_tp = np.sum([item[4] for item in eval_result_items])

        iou = float(total_tp) / (total_tp + total_fp + total_fn)

        torch.set_grad_enabled(True)
        return iou, eval_result_items        
            
            
            
            

            

    
    
            

        