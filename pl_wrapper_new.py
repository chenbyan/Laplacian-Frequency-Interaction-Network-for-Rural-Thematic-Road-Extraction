import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall

from model import LFINet, load_config

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets):        
        # 1. 维度适配
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        with torch.cuda.amp.autocast(enabled=False):
            inputs = inputs.float()
            targets = targets.float()
            
            inputs_clamped = torch.clamp(inputs, min=1e-7, max=1.0-1e-7)
            
            bce_loss = self.bce(inputs_clamped, targets)
            
            inputs_flat = inputs_clamped.reshape(-1)
            targets_flat = targets.reshape(-1)
            
            intersection = (inputs_flat * targets_flat).sum()
            dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
            dice_loss = 1 - dice_score
            
            return 0.5 * bce_loss + 0.5 * dice_loss
# ==================================================================


class ComparisonModelPL(pl.LightningModule):
    def __init__(self, model_name, full_config):
        super().__init__()
        self.save_hyperparameters(ignore=['full_config']) 
        self.model_name = model_name
        raw_lr = full_config.get('BASE_LR', full_config.get('learning_rate'))
        
        try:
            self.lr = float(raw_lr)
        except (TypeError, ValueError):
            print(f"⚠️ 无法解析学习率 {raw_lr}，已退回到默认值 1e-4")
            self.lr = 1e-4
        
        # 1. 初始化模型
        if model_name == "LFINet":
            model_params = full_config.get('model', full_config) 
            self.model = LFINet(model_params)
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.criterion = DiceBCELoss()
        
        self.metric_iou = BinaryJaccardIndex(threshold=0.5)
        self.metric_f1 = BinaryF1Score(threshold=0.5)
        self.metric_prec = BinaryPrecision(threshold=0.5)
        self.metric_recall = BinaryRecall(threshold=0.5)

        # ImageNet Normalization Constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        out = self.model(x)
             
        return out

    def training_step(self, batch, batch_idx):
        x = batch['rgb'].permute(0, 3, 1, 2).float() 
        y = batch['road_mask'].unsqueeze(1).float()

        probs = self(x) 
        
        loss = self.criterion(probs, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['rgb'].permute(0, 3, 1, 2).float()
        y = batch['road_mask'].unsqueeze(1).float()

        probs = self(x)
        loss = self.criterion(probs, y)
        
        # Metrics
        self.metric_iou(probs, y)
        self.metric_f1(probs, y)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.metric_iou, on_epoch=True)
        self.log('val_f1', self.metric_f1, on_epoch=True)

        # 可视化
        if batch_idx == 0:
            try:
                n = min(x.shape[0], 4)
                viz_x = x[:n].clone()
                viz_y = y[:n].clone()
                viz_pred = probs[:n].clone()
                
                viz_x = viz_x * self.std + self.mean
                
                images = []
                for i in range(n):
                    img_t = viz_x[i].permute(1, 2, 0).cpu().numpy()
                    gt_t = viz_y[i].squeeze().cpu().numpy()
                    pred_t = viz_pred[i].squeeze().cpu().numpy()
                    
                    images.append(wandb.Image(
                        np.clip(img_t, 0, 1), 
                        masks={
                            "predictions": {"mask_data": (pred_t > 0.5).astype(int), "class_labels": {0: "bg", 1: "road"}},
                            "ground_truth": {"mask_data": gt_t.astype(int), "class_labels": {0: "bg", 1: "road"}}
                        },
                        caption=f"Sample {i}"
                    ))
                self.logger.experiment.log({f"val_samples_{self.model_name}": images})
            except Exception as e:
                pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]



def verify():
    config_path = "config/256_agri_epoch100.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    try:
        # 1. 测试配置加载
        config = load_config(config_path)
        print("✅ 配置文件加载成功")

        # 2. 测试模型初始化并移动到 GPU
        model_name = "dual_t2r"
        pl_model = ComparisonModelPL(model_name=model_name, full_config=config).to(device)
        pl_model.eval() # 设置为评估模式
        print(f"✅ 模型 {model_name} 在 PL Wrapper 中初始化成功并已移至 {device}")

        # 3. 测试前向传播：输入必须 move 到 CUDA
        dummy_input = torch.randn(1, 3, 256, 256).to(device) 
        
        # ⚡ Mamba 算子通常不支持 float16 的 CPU 模拟，必须在 GPU 上跑
        with torch.no_grad():
            output = pl_model.model(dummy_input)
            
        print(f"✅ 前向传播测试成功，输出维度: {output.shape}")

    except Exception as e:
        print(f"❌ 验证失败！错误信息: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()