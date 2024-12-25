import torch

# Dataset Configuration
DATASET_CONFIG = {
    # Basic dataset parameters
    'batch_size': 8,
    'num_workers': 0,
    'timeout': 10,
    'max_attempts': 10,
    'shuffle': True,
    'shuffle_buffer_size': 10000,

    # Text parameters
    'min_text_length': 4,
    'max_text_length': 77,

    # Image parameters
    'image_size': 224,
    'min_image_size': 32,
    'max_image_size': 1024,

    # Augmentation parameters
    'random_crop_scale': (0.9, 1.0),
    'random_crop_ratio': (0.75, 1.333),
    'color_jitter_prob': 0.8,
    'color_jitter_params': {
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.1
    },
    'grayscale_prob': 0.1,
    'gaussian_blur_prob': 0.1,
    'gaussian_blur_kernel': 23,
    'gaussian_blur_sigma': (0.1, 2.0),
    'horizontal_flip_prob': 0.5,

    # Normalization parameters
    'image_mean': [0.48145466, 0.4578275, 0.40821073],
    'image_std': [0.26862954, 0.26130258, 0.27577711],

    # Validation parameters
    'validation_size': 1000,
    'validation_frequency': 1000,
}

# Vision Transformer Configuration
VISION_CONFIG = {
    'image_size': DATASET_CONFIG['image_size'],
    'patch_size': 32,
    'in_channels': 3,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
    'mlp_ratio': 4.0,
    'dropout': 0.1
}

# Text Transformer Configuration
TEXT_CONFIG = {
    'vocab_size': 49408,
    'max_seq_len': 77,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
    'mlp_ratio': 4.0,
    'dropout': 0.1
}

# CLIP Model Configuration
CLIP_CONFIG = {
    'image_size': VISION_CONFIG['image_size'],
    'patch_size': VISION_CONFIG['patch_size'],
    'embed_dim': VISION_CONFIG['embed_dim'],
    'num_layers': VISION_CONFIG['num_layers'],
    'num_heads': VISION_CONFIG['num_heads'],
    'vocab_size': TEXT_CONFIG['vocab_size'],
    'max_seq_len': TEXT_CONFIG['max_seq_len'],
    'temperature': 0.07,
    'dropout': 0.1
}

# Training Configuration
TRAINING_CONFIG = {
    # Basic training parameters
    'learning_rate': 5e-5,
    'weight_decay': 0.2,
    'warmup_steps': 2000,
    'max_epochs': 32,
    'batch_size': DATASET_CONFIG['batch_size'],
    
    # Optimizer parameters
    'betas': (0.9, 0.98),
    'eps': 1e-6,
    
    # Learning rate schedule
    'min_lr': 1e-6,
    'warmup_start_lr': 1e-8,
    
    # Checkpointing
    'checkpoint_dir': 'checkpoints',
    'save_every_n_epochs': 1,
    'keep_n_checkpoints': 3,
    
    # Hardware
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'compile': True,  # Use torch.compile for speedup if available
    
    # Logging
    'log_every_n_steps': 100,
    'tensorboard_dir': 'runs'
}

