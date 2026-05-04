import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import binary_dilation

from utils import load_config
from dataset_agri import SatMapDataset, graph_collate_fn 

from pl_wrapper_new import ComparisonModelPL 
from model import load_config


def calculate_base_metrics(pred_tensor, target_tensor, threshold=0.5):
    
    preds = (pred_tensor > threshold).float()
    targets = target_tensor.unsqueeze(1).float()
    
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    return tp, fp, fn

def calculate_psnr(img1, img2, data_range=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 10 * torch.log10(data_range ** 2 / mse).item()

def calculate_crr(pred_tensor, true_tensor, buffer_pixels=3):
    pred_road = pred_tensor.cpu().numpy().squeeze()
    true_road = true_tensor.cpu().numpy().squeeze()

    if pred_road.ndim == 2: 
        pred_road = pred_road[np.newaxis, ...]
        true_road = true_road[np.newaxis, ...]
        
    total_crr = 0.0
    count = 0
    
    for i in range(pred_road.shape[0]):
        p = (pred_road[i] > 0.5).astype(np.uint8)
        t = (true_road[i] > 0.5).astype(np.uint8)
        
        if np.sum(p) == 0: continue 
        
        true_buffer = binary_dilation(t, iterations=buffer_pixels)
        correct_road = np.logical_and(p, true_buffer)
        
        val = np.sum(correct_road) / np.sum(p)
        total_crr += val
        count += 1
        
    return total_crr, count

def eval_model(model, loader, model_name, device='cuda'):
    print(f"\n🚀 正在评估: {model_name} ...")
    model.eval()
    model.to(device)
    
    # 累加器
    total_tp, total_fp, total_fn = 0, 0, 0
    psnr_list = []
    crr_sum = 0
    crr_count = 0
    
    loop = tqdm(loader)
    
    with torch.no_grad():
        for batch in loop:
            x = batch['rgb'].to(device)
            y = batch['road_mask'].to(device)
            
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2).float()
                
            # 2. 模型推理
            if isinstance(model, ComparisonModelPL):
                logits = model(x)
                pred = torch.sigmoid(logits)
            else:
                logits = model(x)
                pred = torch.sigmoid(logits)

            tp, fp, fn = calculate_base_metrics(pred, y)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            y_expanded = y.unsqueeze(1).float()
            psnr_list.append(calculate_psnr(pred, y_expanded))
            
            c_val, c_cnt = calculate_crr(pred, y)
            crr_sum += c_val
            crr_count += c_cnt

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_crr = crr_sum / (crr_count + 1e-6)

    print(f"[{model_name}] 结果:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  IoU      : {iou:.4f}")
    print(f"  PSNR     : {avg_psnr:.4f}")
    print(f"  CRR      : {avg_crr:.4f}")
    
    return {
        'Precision': precision, 
        'Recall': recall, 
        'F1': f1, 
        'IoU': iou, 
        'PSNR': avg_psnr, 
        'CRR': avg_crr
    }

if __name__ == "__main__":
    config_path = "config/256_agri_epoch100.yaml"
    sam_config_obj = load_config(config_path)
    
    print("正在加载验证集 (所有模型共用)...")
    val_ds = SatMapDataset(sam_config_obj, is_train=False, dev_run=False) 
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=graph_collate_fn
    )
    
    results = {}

    LFINet_ckpt = "./20260104_140808_ewlfip_LFINet_best_epoch=099_val_iou=0.8262.ckpt"    
    if os.path.exists(LFINet_ckpt):
        try:
            config_path = "config/256_agri_epoch100.yaml"
            full_config = load_config(config_path)
            
            # 显式传入 model_name 和 full_config
            model = ComparisonModelPL.load_from_checkpoint(
                checkpoint_path=LFINet_ckpt,
                model_name='LFINet',
                full_config=full_config
            )
            
            results['LFINet'] = eval_model(model, val_loader, "LFINet")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"LFINet Error: {e}")
            import traceback
            traceback.print_exc() # 打印具体错误堆栈以便调试


    # ================= 打印最终大表格 =================
    print("\n" + "="*85)
    print(f"{'Model':<15} {'F1':<10} {'IoU':<10} {'Prec':<10} {'Recall':<10} {'CRR':<10} {'PSNR':<10}")
    print("-" * 85)
    for name, res in results.items():
        print(f"{name:<15} {res['F1']:.4f}     {res['IoU']:.4f}     {res['Precision']:.4f}     {res['Recall']:.4f}     {res['CRR']:.4f}     {res['PSNR']:.4f}")
    print("="*85)