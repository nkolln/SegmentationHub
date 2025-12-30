import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.logger import Logger
import wandb
from torchmetrics import JaccardIndex
import numpy as np
from src.training.losses import CombinedLoss,DiceLoss
from src.training.focal_loss import FocalLoss
from torch.optim.lr_scheduler import OneCycleLR

class Trainer:
    """
    Trainer class for managing the training and validation loops.
    """
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        # Handle class weights if provided, convert to tensor
        class_weights = config['loss'].get('class_weights', None)
        if class_weights:
             class_weights = torch.tensor(class_weights).float().to(self.device)
        # # class_weights = compute_class_weights(train_dataset)
        # class_sample_count = np.unique(train_loader.dataset.targets, return_counts=True)[1]
        # weight = 1. / class_sample_count
        # samples_weight = weight[target]
        # class_weights = torch.from_numpy(samples_weight)

        loss_type = config['loss'].get('type', 'cross_entropy')
        if loss_type == 'combined':
            self.criterion = CombinedLoss(weight=class_weights)
        elif loss_type == 'focal':
            self.criterion = FocalLoss()
        elif loss_type == 'dice':
            self.criterion = DiceLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            
        self.logger = Logger(config)
        self.global_step = 0
        
        # Metrics
        self.num_classes = config['model']['num_classes']
        self.iou_metric = JaccardIndex(task="multiclass", num_classes=self.num_classes).to(self.device)
        
        # Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=6e-5,
            epochs=config['training']['num_epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,  # Warmup for 30% of training
        )

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            # ... existing loop ...
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks.long())  # Ensure masks are long
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Calculate gradient norm
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            self.optimizer.step()
            
            # Logging
            if batch_idx % self.config['logging']['log_every_n_steps'] == 0:
                self.logger.log_metrics({
                    'train_loss': loss.item(),
                    'train_grad_norm': total_norm,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step)
            
            self.global_step += 1
            pbar.set_postfix({'loss': loss.item()})
            
        # Step scheduler at epoch end
        self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        self.iou_metric.reset()
        
        # Select one batch for visualization (first batch)
        vis_images, vis_masks, vis_preds = None, None, None
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.long())
                val_loss += loss.item()
                
                # Update IoU metric
                preds = torch.argmax(outputs, dim=1)
                self.iou_metric.update(preds, masks)
                
                # Capture variables for visualization from the first batch
                if batch_idx == 0:
                    vis_images = images.cpu()
                    vis_masks = masks.cpu()
                    vis_preds = preds.cpu()
        
        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        mean_iou = self.iou_metric.compute().item()
        
        metrics = {
            'val_loss': avg_val_loss,
            'val_iou': mean_iou
        }
        
        # Log visualization (only if wandb is active)
        if self.config['logging']['use_wandb'] and vis_images is not None:
            # Take the first image in the batch
            image = vis_images[0] # Tensor (3, H, W)
            
            # Unnormalize: (image * std) + mean
            mean = torch.tensor([0.485, 0.456, 0.406]).to(image.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(image.device).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            gt_mask = vis_masks[0].numpy()
            pred_mask = vis_preds[0].numpy()
            
            # Create interactive mask overlay
            class_labels = {i: f"Class {i}" for i in range(self.num_classes)}
            
            # Debug: Print stats about what we are visualizing
            print(f"DEBUG Vis: GT values: {np.unique(gt_mask)}, Pred values: {np.unique(pred_mask)}")
            
            wandb_image = wandb.Image(image_np, masks={
                "predictions": {
                    "mask_data": pred_mask,
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": gt_mask,
                    "class_labels": class_labels
                }
            }, caption=f"Epoch {epoch} Prediction")
            
            metrics['val_prediction'] = wandb_image

        self.logger.log_metrics(metrics, step=self.global_step)
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Val IoU: {mean_iou:.4f}")

    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.train_epoch(epoch)
            self.validate(epoch)
            # Save checkpoint logic here (omitted for brevity)
        
        self.logger.finish()
