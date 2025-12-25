import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.logger import Logger

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

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.logger = Logger(config)
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks.long())  # Ensure masks are long
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            if batch_idx % self.config['logging']['log_every_n_steps'] == 0:
                self.logger.log_metrics({'train_loss': loss.item()}, step=self.global_step)
            
            self.global_step += 1
            pbar.set_postfix({'loss': loss.item()})

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.long())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        self.logger.log_metrics({'val_loss': avg_val_loss}, step=self.global_step)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss}")

    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.train_epoch(epoch)
            self.validate(epoch)
            # Save checkpoint logic here (omitted for brevity)
        
        self.logger.finish()
