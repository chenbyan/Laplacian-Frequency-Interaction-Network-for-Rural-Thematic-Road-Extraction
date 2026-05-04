import os
import torch
import time
import random
import string
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# 引入相关模块
from pl_wrapper_new import ComparisonModelPL
from dataset_agri import SatMapDataset, graph_collate_fn
from utils import load_config

# 生成唯一训练标识（时间戳 + 随机字符串）
def generate_train_id(length=6):
    """生成唯一训练ID：例如 20251202_153045_abc123"""
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{timestamp}_{random_str}"


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        help="Model name: adinknet, hrnet, linknet, spdinknet, t2rnet, setr")
    parser.add_argument("--config", type=str, default="config/256_agri_epoch100.yaml",
                        help="读取参数")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--precision", default=16, help="32 or 16")
    parser.add_argument("--fast_dev_run", default=False, action='store_true')
    parser.add_argument("--dev_run", default=False, action='store_true')
    # 从config读取epochs，移除硬编码
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run

    # 生成唯一训练目录
    root_ckpt_dir = "./lightning_logs/comparison"
    train_id = generate_train_id()
    train_ckpt_root = os.path.join(root_ckpt_dir, args.model, train_id)
    all_ckpt_dir = os.path.join(train_ckpt_root, "all_ckpt")
    best_ckpt_dir = os.path.join(train_ckpt_root, "best_ckpt")
    os.makedirs(all_ckpt_dir, exist_ok=True)
    os.makedirs(best_ckpt_dir, exist_ok=True)
    print(f"📌 本次训练CKPT保存路径：{train_ckpt_root}")

    # 初始化模型
    print(f"🚀 初始化对比模型: {args.model}")
    # 在初始化模型前添加
    torch.backends.cudnn.benchmark = True  # 固定输入形状时加速
    torch.backends.cudnn.enabled = True
    model = ComparisonModelPL(model_name=args.model, full_config=config)

    # 准备数据
    print("加载数据集...")
    train_ds = SatMapDataset(config, is_train=True, dev_run=dev_run)
    val_ds = SatMapDataset(config, is_train=False, dev_run=dev_run)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,  # 从config读取工作进程数
        pin_memory=True,
        collate_fn=graph_collate_fn 
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn
    )

    # 设置WandB
    wandb_logger = WandbLogger(
        project="sam_road_comparison",
        name=f"{args.model}_{train_id}",  # 加入训练ID方便追踪
        config={**config, "model": args.model},  # 合并配置和模型名
        mode='disabled' if dev_run else None
    )

    # 回调1：保存所有epoch的checkpoint（每5个epoch一次）
    checkpoint_all = ModelCheckpoint(
        dirpath=all_ckpt_dir,
        filename=f"{args.model}_all_{{epoch:03d}}_{{step}}",
        every_n_epochs=5,
        save_top_k=-1,
        save_weights_only=False,
        save_on_train_epoch_end=True
    )

    # 回调2：保存最优模型（监控IoU指标）
    checkpoint_best = ModelCheckpoint(
        dirpath=best_ckpt_dir,
        filename=f"{args.model}_best_{{epoch:03d}}_{{val_iou:.4f}}",
        monitor="val_iou",  # 确保与模型中log的指标名一致
        mode="max",
        save_top_k=3,  # 保存top3最优模型
        save_weights_only=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')  # 更精细的学习率监控

    # 训练器配置
    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,  # 从config读取epochs
        accelerator="auto",  # 自动选择加速器
        devices="auto",      # 自动选择设备
        callbacks=[checkpoint_all, checkpoint_best, lr_monitor],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,  # 增加sanity check
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        # 启用cudnn优化
        enable_model_summary=True,
    )

    # 开始训练
    print(f"开始训练 {args.model} ...")
    # 加载配置后添加
    print(f"加载的配置文件: {args.config}")
    print(f"配置的训练轮次: {config.TRAIN_EPOCHS}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    # 打印最终保存路径
    print(f"\n✅ 训练完成！CKPT保存路径：")
    print(f"   全量CKPT：{all_ckpt_dir}")
    print(f"   最优CKPT：{best_ckpt_dir}")


if __name__ == "__main__":
    # 设置临时目录
    os.environ["WANDB_TEMP"] = os.path.join(os.getcwd(), "wandb_tmp")
    os.makedirs(os.environ["WANDB_TEMP"], exist_ok=True)
    os.environ["TMPDIR"] = os.path.join(os.getcwd(), "multiprocess_tmp")
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    
    main()