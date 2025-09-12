#!/usr/bin/env python3
"""
Enhanced training script for 2D StarDist model with NuInsSeg dataset
Supports YAML configuration files and command line arguments
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}

def merge_configs(file_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge file config with command line arguments"""
    config = file_config.copy()
    
    # Override with command line arguments if provided
    if hasattr(args, 'data_dir') and args.data_dir:
        config.setdefault('data', {})['data_dir'] = args.data_dir
    if hasattr(args, 'model_name') and args.model_name:
        config.setdefault('model', {})['name'] = args.model_name
    if hasattr(args, 'epochs') and args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if hasattr(args, 'wandb') and args.wandb is not None:
        config.setdefault('logging', {})['use_wandb'] = args.wandb
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Train StarDist model on NuInsSeg dataset")
    
    # Configuration file
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    
    # Override arguments (optional)
    parser.add_argument("--data_dir", type=str, help="Data directory (overrides config)")
    parser.add_argument("--model_name", type=str, help="Model name (overrides config)")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb (overrides config)")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable wandb")
    parser.set_defaults(wandb=None)
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        file_config = load_config(args.config)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
        file_config = {}
    
    # Merge configurations
    config = merge_configs(file_config, args)
    
    # Set default values if not in config
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    logging_config = config.get('logging', {})
    hardware_config = config.get('hardware', {})
    
    # Validate required parameters
    if 'data_dir' not in data_config:
        logger.error("data_dir must be specified in config file or command line")
        return 1
    
    # Import here to avoid long startup time if config is invalid
    from train_stardist_nuinsseg import (
        NuInsSegDataLoader, StarDistTrainer, create_augmenter
    )
    import tensorflow as tf
    from stardist.models import Config2D
    from stardist import gputools_available
    
    # Setup GPU
    gpu_id = hardware_config.get('gpu_id', 0)
    gpu_memory_limit = hardware_config.get('gpu_memory_limit')
    
    if tf.config.list_physical_devices('GPU'):
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_id < len(gpu_devices):
            tf.config.set_visible_devices(gpu_devices[gpu_id], 'GPU')
            if gpu_memory_limit:
                tf.config.experimental.set_memory_growth(gpu_devices[gpu_id], True)
                tf.config.experimental.set_memory_limit(
                    gpu_devices[gpu_id], gpu_memory_limit
                )
            logger.info(f"Using GPU {gpu_id}")
        else:
            logger.warning(f"GPU {gpu_id} not available, using CPU")
    else:
        logger.info("No GPU available, using CPU")
    
    # Enable mixed precision if requested
    if hardware_config.get('use_mixed_precision', False):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Enabled mixed precision training")
    
    # Load data
    logger.info("Loading NuInsSeg dataset...")
    data_loader = NuInsSegDataLoader(
        data_config['data_dir'], 
        data_config.get('tissues')
    )
    X, Y = data_loader.load_data(data_config.get('max_images_per_tissue'))
    
    if len(X) == 0:
        logger.error("No data loaded!")
        return 1
    
    # Create configuration
    n_channel = 1 if X[0].ndim == 2 else (1 if len(X[0].shape) == 2 else X[0].shape[-1])
    
    stardist_config = Config2D(
        n_rays=model_config.get('n_rays', 32),
        grid=tuple(model_config.get('grid', [2, 2])),
        n_channel_in=n_channel,
        use_gpu=gputools_available(),
        train_epochs=training_config.get('epochs', 400),
        train_steps_per_epoch=training_config.get('steps_per_epoch', 200),
        train_batch_size=training_config.get('batch_size', 4),
        train_learning_rate=training_config.get('learning_rate', 0.0003),
        train_patch_size=tuple(model_config.get('patch_size', [256, 256])),
        train_tensorboard=logging_config.get('tensorboard', True),
        train_shape_completion=training_config.get('shape_completion', False),
        train_completion_crop=training_config.get('completion_crop', 32),
    )
    
    logger.info(f"Model configuration:")
    logger.info(f"  Number of rays: {stardist_config.n_rays}")
    logger.info(f"  Grid: {stardist_config.grid}")
    logger.info(f"  Input channels: {stardist_config.n_channel_in}")
    logger.info(f"  Batch size: {stardist_config.train_batch_size}")
    logger.info(f"  Learning rate: {stardist_config.train_learning_rate}")
    logger.info(f"  Patch size: {stardist_config.train_patch_size}")
    
    # Initialize trainer
    trainer = StarDistTrainer(
        stardist_config, 
        model_config.get('name', 'stardist_nuinsseg'),
        model_config.get('save_dir', 'models')
    )
    
    # Prepare data
    X_train, Y_train, X_val, Y_val = trainer.prepare_data(
        X, Y, data_config.get('val_split', 0.15)
    )
    
    # Train model
    trainer.train(
        X_train, Y_train, X_val, Y_val,
        epochs=training_config.get('epochs', 400),
        steps_per_epoch=training_config.get('steps_per_epoch', 200),
        use_wandb=logging_config.get('use_wandb', False),
        wandb_project=logging_config.get('wandb_project', 'stardist-nuinsseg')
    )
    
    # Optimize thresholds if requested
    evaluation_config = config.get('evaluation', {})
    if evaluation_config.get('optimize_thresholds', True):
        trainer.optimize_thresholds(X_val, Y_val)
    
    # Evaluate model
    logger.info("Evaluating model...")
    iou_thresholds = evaluation_config.get('iou_thresholds', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    trainer.evaluate(X_val, Y_val, iou_thresholds)
    
    model_path = Path(model_config.get('save_dir', 'models')) / model_config.get('name', 'stardist_nuinsseg')
    logger.info(f"Training completed! Model saved in: {model_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
