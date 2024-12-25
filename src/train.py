import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime
import logging

from model import CLIP
from dataset import LaionDataset
from config import CLIP_CONFIG, TRAINING_CONFIG, DATASET_CONFIG

class CLIPTrainer:
    def __init__(
            self,
            model,
            train_dataset,
            val_dataset=None,
            config=TRAINING_CONFIG
    ):
        self.config = config
        self.device = config['device']
        
        # Move model to device and potentially compile it
        self.model = model.to(self.device)
        if config['compile'] and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(config['tensorboard_dir'])
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=config['betas'],
            eps=config['eps']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = self._setup_scheduler()

    def _setup_scheduler(self):
        total_steps = len(self.train_dataset) * self.config['max_epochs']
        return CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config['min_lr'],
            warmup_start_lr=self.config['warmup_start_lr']
        )

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if specified
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataset, desc=f'Epoch {epoch}')
        
        for batch in progress_bar:
            # Move batch to device
            images = batch['image'].to(self.device)
            text = batch['input_ids'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images, text)
            loss = self.model.contrastive_loss(logits)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            wandb.log({
                'train_batch_loss': loss.item(),
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
        
        epoch_loss = total_loss / num_batches
        return epoch_loss

    def validate(self):
        if self.val_dataset is None:
            return None
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataset, desc='Validation'):
                images = batch['image'].to(self.device)
                text = batch['input_ids'].to(self.device)
                
                logits = self.model(images, text)
                loss = self.model.contrastive_loss(logits)
                
                total_loss += loss.item()
                num_batches += 1
        
        val_loss = total_loss / num_batches
        return val_loss

    def train(self):
        best_loss = float('inf')
        
        for epoch in range(self.config['max_epochs']):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_loss = self.validate()
            
            # Log metrics
            metrics = {
                'train_epoch_loss': train_loss,
                'epoch': epoch
            }
            if val_loss is not None:
                metrics['val_loss'] = val_loss
            wandb.log(metrics)
            
            # Save checkpoint
            is_best = val_loss is not None and val_loss < best_loss
            self.save_checkpoint(epoch, train_loss, is_best)
            
            if is_best:
                best_loss = val_loss

def main():
    try:
        # Initialize wandb with more configuration
        wandb.init(
            project="clip-training",
            config={
                "model_config": CLIP_CONFIG,
                "training_config": TRAINING_CONFIG,
                "dataset_config": DATASET_CONFIG
            },
            name=f"clip_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Create datasets with error handling
        try:
            train_dataset = LaionDataset(split="train", is_validation=False)
            val_dataset = LaionDataset(split="train", is_validation=True)
        except Exception as e:
            logging.error(f"Failed to create datasets: {str(e)}")
            raise
            
        # Create model
        model = CLIP()
        logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Initialize trainer
        trainer = CLIPTrainer(
            model=model,
            train_dataset=train_dataset.get_dataloader(),
            val_dataset=val_dataset.get_dataloader()
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    main()