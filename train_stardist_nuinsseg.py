#!/usr/bin/env python3
"""
Training script for 2D StarDist models with NuInsSeg dataset

This script handles data loading, augmentation, training with wandb logging,
checkpointing, threshold optimization, and automatic dataset management.

Usage:
    python train_stardist_nuinsseg.py config.yaml
    # or if config.yaml is in configuration directory "configs/":
    python train_stardist_nuinsseg.py configs/config.yaml

    # or use as function:
    from train_stardist_nuinsseg import train_from_config
    train_from_config("configs/config.yaml")
"""

import os
import sys
import numpy as np
from pathlib import Path
from glob import glob
from fnmatch import fnmatch
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any, Union
import json
import pickle
import warnings
import yaml
import shutil
from datetime import datetime

from tifffile import imread
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

import tensorflow as tf
from csbdeep.utils import normalize
from stardist import fill_label_holes, calculate_extents, gputools_available, random_label_cmap
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D

# Weights and Biases for logging
try:
    import wandb
    from wandb.integration.keras import WandbCallback
    WANDB_AVAILABLE = True
    print("\nWandb imported successfully")

    try:
        api = wandb.Api()
        user = api.viewer
        print(f"Wandb user authenticated: {user}")
    except Exception as e:
        print(f"Wandb authentication issue: {e}")
        print("Run 'wandb login' to authenticate")
        
except ImportError:
    WANDB_AVAILABLE = False
    print("Wandb not available. Install with 'pip install wandb' for experiment tracking.")

np.random.seed(42)
tf.random.set_seed(42)

class NuInsSegDataLoader:
    """PyTorch-style data loader for NuInsSeg dataset"""
    
    def __init__(self, data_dir: str, tissue_types: Optional[Union[List[str], str]] = None, 
                 max_images_per_tissue: Optional[int] = None):
        """
        Initialize NuInsSeg data loader
        
        Args:
            data_dir: Path to NuInsSeg dataset directory
            tissue_types: Tissue selection options:
                         - None: Use all available tissues
                         - List of strings: Can include exact names and glob patterns
                           Examples: ["human brain"], ["human*"], ["human*", "mouse liver"]
                         - Single string: Single tissue name or glob pattern
            max_images_per_tissue: Maximum number of images to load per tissue type
        """
        self.data_dir = Path(data_dir)
        self.max_images_per_tissue = max_images_per_tissue
        
        all_tissues = [d.name for d in self.data_dir.iterdir() 
                      if d.is_dir() and 'zip' not in d.name.lower()]
        
        self.tissue_types = self._resolve_tissue_patterns(tissue_types, all_tissues)
            
        print(f"\nAvailable tissues ({len(all_tissues)}): \n{sorted(all_tissues)}")
        print(f"\nSelected tissues ({len(self.tissue_types)}): \n{sorted(self.tissue_types)}\n")
        
        self.images, self.labels, self.file_paths = self._load_all_data()
        print(f"\nDataset initialized with {len(self.images)} image-label pairs")
    
    def _resolve_tissue_patterns(self, tissue_types: Optional[Union[List[str], str]], 
                                all_tissues: List[str]) -> List[str]:
        """
        Resolve tissue patterns (glob patterns and exact matches) to actual tissue names
        
        Args:
            tissue_types: Input tissue specification (None, string, or list of strings)
            all_tissues: List of all available tissue names
            
        Returns:
            List of resolved tissue names
        """
        if tissue_types is None:
            return all_tissues
        
        if isinstance(tissue_types, str):
            tissue_types = [tissue_types]
        
        selected_tissues = set()
        
        for pattern in tissue_types:
            if pattern in all_tissues:
                selected_tissues.add(pattern)
            else:
                matches = [tissue for tissue in all_tissues if fnmatch(tissue, pattern)]
                if matches:
                    selected_tissues.update(matches)
        
        return sorted(list(selected_tissues))

    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sample by index
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image, label) arrays
        """
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        return self.images[idx], self.labels[idx]
        
    def _load_all_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, str]]]:
        """
        Load all images and corresponding labels with file tracking and shape validation
            
        Returns:
            Tuple of (images, labels, file_paths) lists
        """
        images = []
        labels = []
        file_paths = []
        
        for tissue in self.tissue_types:
            tissue_dir = self.data_dir / tissue
            
            img_dir = tissue_dir / "tissue images"
            lbl_dir = tissue_dir / "label masks modify"
            
            if not (img_dir.exists() and lbl_dir.exists()):
                print(f"Warning: Missing directories for tissue {tissue}, skipping...")
                continue
                
            img_files = sorted(glob(str(img_dir / "*.png")))
            lbl_files = sorted(glob(str(lbl_dir / "*.tif")))
            
            if len(img_files) != len(lbl_files):
                print(f"Warning: Mismatch in number of images and labels for {tissue}")
                min_len = min(len(img_files), len(lbl_files))
                img_files = img_files[:min_len]
                lbl_files = lbl_files[:min_len]
            
            if self.max_images_per_tissue:
                img_files = img_files[:self.max_images_per_tissue]
                lbl_files = lbl_files[:self.max_images_per_tissue]
            
            print(f"Loading {len(img_files)} images from {tissue}...")
            
            for img_file, lbl_file in tqdm(zip(img_files, lbl_files), 
                                         desc=f"Loading {tissue}", 
                                         total=len(img_files)):
                try:
                    img = imageio.imread(img_file)
                    original_img_shape = img.shape
                    
                    lbl = imread(lbl_file)
                    original_lbl_shape = lbl.shape
                    
                    if lbl.dtype != np.uint16:
                        lbl = lbl.astype(np.uint16)
                    
                    # Shape validation and correction
                    shape_corrected = False
                    
                    if img.shape[:2] != lbl.shape[:2]:
                        print(f"ERROR: First 2 dimensions don't match:")
                        print(f"  Image file: {img_file}")
                        print(f"  Label file: {lbl_file}")
                        print(f"  Image shape: {original_img_shape}")
                        print(f"  Label shape: {original_lbl_shape}")
                        continue
                    
                    if img.ndim == 3: # RGB
                        if img.shape[2] == 3:
                            pass
                        elif img.shape[2] > 3: # additional channels
                            img = img[..., :3]
                            shape_corrected = True
                            print(f"Removed {original_img_shape[2] - 3} zero channels from {Path(img_file).name}")
                        else:
                            raise ValueError(f"Image has only {img.shape[2]} channels, expected 3 (RGB): {Path(img_file).name}")
                    # elif img.ndim == 2:
                    #     # Convert grayscale to RGB by repeating the channel
                    #     img = np.stack([img, img, img], axis=-1)
                    #     shape_corrected = True
                    #     print(f"Converted grayscale to RGB for {Path(img_file).name}")
                    else:
                        raise ValueError(f"Unexpected image dimensions: {img.shape} for {Path(img_file).name}")
                    
                    # Final validation - image shape should be (512, 512, 3)
                    if img.shape != (512, 512, 3):
                        print(f"ERROR: Final image shape is not (512, 512, 3):")
                        print(f"  File: {Path(img_file).name}")
                        print(f"  Shape: {img.shape}")
                        continue
                        
                    images.append(img)
                    labels.append(lbl)
                    
                    # Store file information
                    file_paths.append({
                        'dataset_index': len(images) - 1,
                        'image_path': str(img_file),
                        'label_path': str(lbl_file),
                        'tissue': tissue,
                        'image_name': Path(img_file).name,
                        'label_name': Path(lbl_file).name,
                        'original_img_shape': original_img_shape,
                        'original_lbl_shape': original_lbl_shape,
                        'final_shape': img.shape,
                        'shape_corrected': shape_corrected
                    })
                    
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
        
        return images, labels, file_paths
    
    def get_file_info(self, dataset_index: int) -> Dict[str, Any]:
        """Get file information for any dataset index"""
        if dataset_index >= len(self.file_paths):
            raise IndexError(f"Index {dataset_index} out of range for dataset of size {len(self.file_paths)}")
        return self.file_paths[dataset_index]
    
    def get_subset(self, indices: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Get a subset of the dataset by indices
        
        Args:
            indices: List of indices to include in subset
            
        Returns:
            Tuple of (images, labels, file_paths) lists for the subset
        """
        subset_images = [self.images[i] for i in indices]
        subset_labels = [self.labels[i] for i in indices]
        subset_file_paths = [self.file_paths[i] for i in indices]
        return subset_images, subset_labels, subset_file_paths

def create_augmenter():
    """Create data augmentation function"""
    
    def random_fliprot(img, mask): 
        """Random flips and rotations"""
        assert img.ndim >= mask.ndim
        axes = tuple(range(mask.ndim))
        perm = tuple(np.random.permutation(axes))
        img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
        mask = mask.transpose(perm) 
        for ax in axes: 
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask 

    def random_intensity_change(img):
        """Random intensity variations"""
        img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
        return img

    def augmenter(x, y):
        """Main augmentation function"""
        x, y = random_fliprot(x, y)
        x = random_intensity_change(x)
        sig = 0.02 * np.random.uniform(0, 1)
        x = x + sig * np.random.normal(0, 1, x.shape)
        return x, y
    
    return augmenter

class PredictionVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, X_data, Y_data, file_paths=None, log_every_n_epochs=20, num_samples=4, wandb_key="val_predictions"):
        super().__init__()
        self.stardist_model = model
        self.X_data = X_data
        self.Y_data = Y_data
        self.file_paths = file_paths
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = min(num_samples, len(X_data))
        self.wandb_key = wandb_key
        
        np.random.seed(42)
        self.sample_indices = np.random.choice(len(X_data), self.num_samples, replace=False)
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every_n_epochs == 0:
            self.log_predictions(epoch + 1)
    
    def log_predictions(self, epoch):
        images_to_log = []
        
        for i, idx in enumerate(self.sample_indices):
            image = self.X_data[idx]
            gt_mask = self.Y_data[idx]
            
            pred_mask, _ = self.stardist_model.predict_instances(image)
            
            fig = self.create_overlay_visualization(image, gt_mask, pred_mask, idx)
            
            wandb_image = wandb.Image(fig, caption=f"Sample {idx} - Epoch {epoch}")
            images_to_log.append(wandb_image)
            
            plt.close(fig)
        
        wandb.log({
            self.wandb_key: images_to_log,
            "epoch": epoch
        })
    
    def create_overlay_visualization(self, image, gt_mask, pred_mask, sample_idx):
        """Create a 3-panel visualization: original, ground truth overlay, prediction overlay"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get filename for title if file paths are available
        filename = f'Sample {sample_idx}'
        if self.file_paths and sample_idx < len(self.file_paths):
            file_info = self.file_paths[sample_idx]
            filename = file_info.get('image_name', f'Sample {sample_idx}')
            tissue = file_info.get('tissue', '')
            # if tissue:
            #     filename = f"{tissue}: {filename}"
        
        fig.suptitle(f'{filename}', fontsize=14, fontweight='bold')
        
        display_image = np.clip(image, 0, 1)
        
        # Panel 1: Original image
        axes[0].imshow(display_image)
        axes[0].set_title(f"Image (Sample {sample_idx})")
        axes[0].axis('off')
        
        # Panel 2: Ground truth overlay
        axes[1].imshow(display_image)
        gt_overlay = self.create_mask_overlay(gt_mask)
        axes[1].imshow(gt_overlay, alpha=0.5)
        axes[1].set_title(f"GT (n = {np.max(gt_mask)})")
        axes[1].axis('off')
        
        # Panel 3: Prediction overlay
        axes[2].imshow(display_image)
        pred_overlay = self.create_mask_overlay(pred_mask)
        axes[2].imshow(pred_overlay, alpha=0.5)
        axes[2].set_title(f"Prediction (n = {np.max(pred_mask)})")
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_mask_overlay(self, mask):
        """Create an instance segmentation mask color overlay with transparent background"""
        if np.max(mask) == 0:
            return np.zeros((*mask.shape, 4))
        
        lbl_cmap = random_label_cmap()
        
        colored_mask = lbl_cmap(mask)[:, :, :4]
        colored_mask[mask == 0, 3] = 0  # transparent background pixels for alpha = 0
        return colored_mask

class StarDistTrainer:
    """StarDist model trainer with wandb integration"""
    
    def __init__(self, config: Config2D, model_name: str, model_dir: str = "models"):
        """
        Initialize trainer
        
        Args:
            config: StarDist configuration
            model_name: Name for the model
            model_dir: Directory to save models
        """
        self.config = config
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = StarDist2D(config, name=model_name, basedir=str(self.model_dir))
        
        self.history = None
        
    def prepare_data(self, dataset: NuInsSegDataLoader, 
                    train_split: float = 0.7, val_split: float = 0.15, test_split: float = 0.15) -> Tuple[List, List, List, List, List, List]:
        """
        Prepare training, validation, and test data with 3-way split
        
        Args:
            dataset: NuInsSegDataLoader instance
            train_split: Fraction of data to use for training
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, Y_train, X_val, Y_val, X_test, Y_test)
        """
        total_samples = len(dataset)
        
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
        
        n_train = int(train_split * total_samples)
        n_val = int(val_split * total_samples)
        n_test = total_samples - n_train - n_val
        
        rng = np.random.RandomState(42)
        indices = rng.permutation(total_samples)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        X_train_raw, Y_train_raw, train_file_paths = dataset.get_subset(train_indices.tolist())
        X_val_raw, Y_val_raw, val_file_paths = dataset.get_subset(val_indices.tolist())
        X_test_raw, Y_test_raw, test_file_paths = dataset.get_subset(test_indices.tolist())
        
        self.train_file_paths = train_file_paths
        self.val_file_paths = val_file_paths
        self.test_file_paths = test_file_paths
        
        print("\nNormalizing images...")
        axis_norm = (0, 1)
        
        X_train = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X_train_raw, desc="Training")]
        X_val = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X_val_raw, desc="Validation")]
        X_test = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X_test_raw, desc="Test")]
        
        print("\nFilling label holes...")
        Y_train = [fill_label_holes(y) for y in tqdm(Y_train_raw, desc="Training")]
        Y_val = [fill_label_holes(y) for y in tqdm(Y_val_raw, desc="Validation")]
        Y_test = [fill_label_holes(y) for y in tqdm(Y_test_raw, desc="Test")]
        
        print(f'\nDataset split summary:')
        print(f'- Total samples:      {total_samples}')
        print(f'- Training:           {len(X_train)} ({len(X_train)/total_samples:.1%})')
        print(f'- Validation:         {len(X_val)} ({len(X_val)/total_samples:.1%})')
        print(f'- Test:               {len(X_test)} ({len(X_test)/total_samples:.1%})\n')
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    def get_split_file_info(self, split_type: str, split_index: int) -> Dict[str, Any]:
        """Get file information for a specific sample in a split"""
        if split_type == 'train':
            file_paths = self.train_file_paths
        elif split_type == 'val':
            file_paths = self.val_file_paths
        elif split_type == 'test':
            file_paths = self.test_file_paths
        else:
            raise ValueError(f"split_type must be 'train', 'val', or 'test', got {split_type}")
            
        if split_index >= len(file_paths):
            raise IndexError(f"Index {split_index} out of range for {split_type} split of size {len(file_paths)}")
            
        return file_paths[split_index]
    
    def check_field_of_view(self, Y: List[np.ndarray]):
        """Check if network field of view is adequate"""
        print("\nChecking field of view...")
        median_size = calculate_extents(Y, np.median)
        fov = np.array(self.model._axes_tile_overlap('YX'))
        print(f"\nMedian object size:       {median_size}")
        print(f"Network field of view:    {fov}")
        if any(median_size > fov):
            warnings.warn("Median object size larger than field of view of the neural network.")
            return False
        return True
    
    def train(self, X_train: List[np.ndarray], Y_train: List[np.ndarray],
              X_val: List[np.ndarray], Y_val: List[np.ndarray],
              epochs: int = 200, steps_per_epoch: int = None,
              use_wandb: bool = True, wandb_project: str = "stardist-nuinsseg",
              wandb_entity: str = None, wandb_run_name: str = None, wandb_tags: List[str] = None, 
              log_pred_every_n_epochs: int = 20, num_val_samples_to_log: int = 4,
              early_stopping: bool = True, **kwargs):
        """
        Train the StarDist model
        
        Args:
            X_train, Y_train: Training data
            X_val, Y_val: Validation data
            epochs: Number of training epochs
            steps_per_epoch: Steps per epoch (if None, calculated from data size and batch size)
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
            wandb_entity: Wandb entity (team) name
            wandb_run_name: Custom name for the wandb run (if None, uses model_name)
            wandb_tags: List of tags to add to the wandb run for organization
            log_pred_every_n_epochs: Frequency of logging prediction visualizations
            num_val_samples_to_log: Number of validation samples to log predictions for
            early_stopping: Whether to apply an early stopping Callback based on validation loss
            **kwargs: Additional arguments for model.train()
        """
        
        if steps_per_epoch is None:
            batch_size = self.config.train_batch_size
            steps_per_epoch = max(1, len(X_train) // batch_size)
            # print(f"\nCalculated steps_per_epoch: {steps_per_epoch} (from # samples / batch_size: {len(X_train)} / {batch_size})")
        
        if use_wandb and WANDB_AVAILABLE:
            run_name = wandb_run_name if wandb_run_name is not None else self.model_name
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                tags=wandb_tags,
                config={
                    "model_name": self.model_name,
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "n_rays": self.config.n_rays,
                    "grid": self.config.grid,
                    "patch_size": self.config.train_patch_size,
                    "batch_size": self.config.train_batch_size,
                    "learning_rate": self.config.train_learning_rate,
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                }
            )
            
            wandb.config.update(vars(self.config))
            
        self.check_field_of_view(Y_train + Y_val)
    
        augmenter = create_augmenter()
        
        checkpoint_dir = self.model_dir / self.model_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare model for training (creates model.callbacks list)
        self.model.prepare_for_training()
        
        # Wandb callbacks
        wandb_callbacks = []
        if use_wandb and WANDB_AVAILABLE:
            # from wandb.integration.keras import WandbMetricsLogger
            # wandb_metrics_logger = WandbMetricsLogger(
            #     log_freq="epoch",
            # )
            # wandb_callbacks.append(wandb_metrics_logger)

            if log_pred_every_n_epochs is not None and num_val_samples_to_log is not None:
                validation_progress_callback = PredictionVisualizationCallback(
                    model=self.model,
                    X_data=X_val,
                    Y_data=Y_val,
                    file_paths=getattr(self, 'val_file_paths', None),
                    log_every_n_epochs=log_pred_every_n_epochs,
                    num_samples=num_val_samples_to_log
                )
                wandb_callbacks.append(validation_progress_callback)
            else:
                print("Prediction visualization disabled: log_pred_every_n_epochs or num_val_samples_to_log not specified")
        
        # Model checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "epoch_{epoch:03d}.weights.h5"),
            save_weights_only=True,
            save_freq='epoch',
            verbose=1
        )
        self.model.callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        if early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            self.model.callbacks.append(early_stopping_callback)

        if wandb_callbacks is not None:
            self.model.callbacks.extend(wandb_callbacks)
        
        print("\nStarting training...")
        print(f"Training for {epochs} epochs with {steps_per_epoch} steps per epoch")
        
        try:
            self.history = self.model.train(
                X_train, Y_train, 
                validation_data=(X_val, Y_val),
                augmenter=augmenter,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                **kwargs
            )
            
            print("Training completed successfully!")
            
            if use_wandb and WANDB_AVAILABLE and wandb.run is not None and self.history:
                print("\nLogging training history to wandb...")
                
                wandb.define_metric("epoch")
                # wandb.define_metric("loss", step_metric="epoch")
                # wandb.define_metric("val_loss", step_metric="epoch")
                # wandb.define_metric("stardist_metrics/*", step_metric="epoch")
                
                for epoch in range(len(self.history.history['loss'])):
                    log_dict = {}
                    
                    # Log main loss metrics in separate section
                    if 'loss' in self.history.history and epoch < len(self.history.history['loss']):
                        log_dict['loss'] = self.history.history['loss'][epoch]
                    if 'val_loss' in self.history.history and epoch < len(self.history.history['val_loss']):
                        log_dict['val_loss'] = self.history.history['val_loss'][epoch]
                    
                    # Log all other StarDist metrics directly (excluding loss and val_loss)
                    for metric_name, values in self.history.history.items():
                        if metric_name not in ['loss', 'val_loss'] and epoch < len(values):
                            log_dict[f"stardist_metrics/{metric_name}"] = values[epoch]
                    
                    # log_dict['epoch'] = epoch + 1
                    wandb.log(log_dict, step=epoch + 1)
                
                print(f"Logged {len(self.history.history['loss'])} epochs of training metrics to wandb")
                print(f"\nAvailable metrics: {list(self.history.history.keys())}")
            
        except KeyboardInterrupt:
            print("Training interrupted by user!")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
            
        history_file = self.model_dir / self.model_name / "history.pkl"
        if self.history:
            with open(history_file, 'wb') as f:
                pickle.dump(self.history.history, f)
            print(f"\nTraining history saved to {history_file}")

    def save_model(self):
        """Save the trained model in multiple formats with error handling"""
        print("\nSaving fully trained model...")
        model_save_path = self.model_dir / self.model_name
        h5_model_path = model_save_path / f"{self.model_name}.h5"
        keras_model_path = model_save_path / f"{self.model_name}.keras"

        saved_formats = []
        
        # Save in legacy Keras H5 format (.h5)
        try:
            self.model.keras_model.save(str(h5_model_path))
            saved_formats.append(f"Keras H5: {h5_model_path.name}")
        except Exception as e:
            print(f"Warning: Failed to save in H5 format: {e}")
        
        # Save in Keras v3 format (.keras)
        try:
            self.model.keras_model.save(str(keras_model_path))
            saved_formats.append(f"Keras v3: {keras_model_path.name}")
        except Exception as e:
            print(f"Warning: Failed to save in Keras v3 format: {e}")
        
        # Save StarDist model in native format (TFSavedModel)
        try:
            self.model.export_TF()
            saved_formats.append("StarDist TF")
        except Exception as e:
            print(f"Warning: Failed to export StarDist TF format: {e}")
        
        if saved_formats:
            print(f"Model saved to {model_save_path} as:")
            for format_info in saved_formats:
                print(f"  - {format_info}")
        else:
            print("Warning: No model formats could be saved successfully!")

    def optimize_thresholds(self, X_val: List[np.ndarray], Y_val: List[np.ndarray]):
        """Optimize probability and NMS thresholds"""
        print("\nOptimizing thresholds...")
        self.model.optimize_thresholds(X_val, Y_val)
        print("\nThreshold optimization completed!")
        
        print(f"\nOptimized thresholds:")
        print(f"  Probability threshold: {self.model.thresholds.prob}")
        print(f"  NMS threshold: {self.model.thresholds.nms}")
        
        # if WANDB_AVAILABLE and wandb.run is not None:
        #     wandb.run.summary.update({
        #         "thresholds/probability_threshold": self.model.thresholds.prob,
        #         "thresholds/nms_threshold": self.model.thresholds.nms,
        #         "optimization/threshold_optimization_completed": True
        #     })
    
    def log_test_predictions(self, X_test: List[np.ndarray], Y_test: List[np.ndarray], 
                           num_test_samples_to_log: int = 4):
        """Log test predictions with optimized thresholds to wandb"""
        if not (WANDB_AVAILABLE and wandb.run is not None):
            print("Wandb not available, skipping test prediction logging")
            return
        
        test_viz_callback = PredictionVisualizationCallback(
            model=self.model,
            X_data=X_test,
            Y_data=Y_test,
            file_paths=getattr(self, 'test_file_paths', None),
            log_every_n_epochs=1,
            num_samples=num_test_samples_to_log,
            wandb_key="test_predictions"
        )
        
        test_viz_callback.log_predictions("Final")
    
    def evaluate(self, X_data: List[np.ndarray], Y_data: List[np.ndarray], 
                 iou_thresholds: List[float] = None, data_type: str = "validation"):
        """Evaluate model performance"""
        if iou_thresholds is None:
            iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        print(f"Predicting on {data_type} data...")
        Y_pred = []
        for x in tqdm(X_data, desc="Predicting"):
            pred, _ = self.model.predict_instances(
                x, 
                n_tiles=self.model._guess_n_tiles(x), 
                show_tile_progress=False
            )
            Y_pred.append(pred)
        
        print("Computing matching statistics...")
        stats = []
        for tau in tqdm(iou_thresholds, desc="IoU thresholds"):
            stat = matching_dataset(Y_data, Y_pred, thresh=tau, show_progress=False)
            stats.append(stat)
        
        # Print results for IoU = 0.5
        tau_05_idx = iou_thresholds.index(0.5) if 0.5 in iou_thresholds else len(iou_thresholds) // 2
        stat_05 = stats[tau_05_idx]
        print(f"\nEvaluation results on {data_type} data at IoU = {iou_thresholds[tau_05_idx]}:")
        print(f"  Precision: {stat_05.precision:.3f}")
        print(f"  Recall: {stat_05.recall:.3f}")
        print(f"  F1: {stat_05.f1:.3f}")
        print(f"  Accuracy: {stat_05.accuracy:.3f}\n")
        
        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run is not None:
            precision_values, recall_values, f1_values, accuracy_values = zip(*[
                (stat.precision, stat.recall, stat.f1, stat.accuracy) for stat in stats
            ])
            
            eval_line_plot = wandb.plot.line_series(
                xs=iou_thresholds,
                ys=[precision_values, recall_values, f1_values, accuracy_values],
                keys=["Precision", "Recall", "F1_Score", "Accuracy"],
                title=f"{data_type.title()} metrics",
                xname="IoU Threshold"
            )
            
            wandb.log({f"evaluation/{data_type}_metrics_vs_iou": eval_line_plot})
        
        # Save evaluation results
        eval_file = self.model_dir / self.model_name / f"{data_type}_evaluation_results.json"
        eval_data = {
            "iou_thresholds": iou_thresholds,
            "stats": [stat._asdict() for stat in stats]
        }
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        print(f">> {data_type.capitalize()} evaluation results saved to {eval_file}\n")
        
        return stats, Y_pred

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def copy_data_to_scratch(data_config: Dict[str, Any]) -> str:
    """
    Check if dataset exists in scratch directory, if not copy from local directory.
    Always prioritize using scratch directory as it's faster.
    
    Args:
        data_config: Data configuration dictionary
        
    Returns:
        str: Path to the dataset directory to use (preferably scratch)
    """
    local_data_dir = Path(data_config['data_dir'])
    scratch_data_dir = data_config.get('scratch_data_dir', '/scratch')
    copy_to_scratch = data_config.get('copy_to_scratch', True)
    
    # Check if dataset exists on scratch
    if str(scratch_data_dir) == "/scratch" or str(scratch_data_dir) == "/scratch/":
        raise RuntimeError(
            "scratch_dir points to /scratch root directory. Data should always be copied to a dedicated User folder.\n"
            "Please specify a subdirectory under /scratch for dataset storage (e.g., /scratch/user/data/)."
        )
    scratch_data_dir = Path(scratch_data_dir) / "NuInsSeg"
    if scratch_data_dir.exists():
        tissue_dirs = [d for d in scratch_data_dir.iterdir() 
                      if d.is_dir() and 'zip' not in d.name.lower()]
        if tissue_dirs:
            print(f"Using dataset from scratch drive at: {scratch_data_dir}")
            return str(scratch_data_dir)
        else:
            print(f"Scratch dataset directory exists but appears empty: {scratch_data_dir}")
    
    # If not, check if it exists on the given data_dir path and copy to scratch
    if local_data_dir.exists():
        tissue_dirs = [d for d in local_data_dir.iterdir() 
                      if d.is_dir() and 'zip' not in d.name.lower()]
        if tissue_dirs:
            if copy_to_scratch:
                print(f"Found dataset locally at: {local_data_dir}")
                print(f"Copying dataset from {local_data_dir} to scratch at {scratch_data_dir} for faster access...")
                
                scratch_data_dir.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    shutil.copytree(local_data_dir, scratch_data_dir, dirs_exist_ok=True)
                    print(f"Dataset copied successfully to scratch: {scratch_data_dir}")
                    return str(scratch_data_dir)
                except Exception as e:
                    print(f"Warning: Failed to copy dataset to scratch: {e}")
                    print(f"Falling back to using local dataset: {local_data_dir}")
                    return str(local_data_dir)
            else:
                print(f"copy_to_scratch disabled, using local dataset: {local_data_dir}")
                return str(local_data_dir)
        else:
            print(f"Local dataset directory exists but appears empty: {local_data_dir}")
    
    raise FileNotFoundError(
        f"NuInsSeg dataset not found in any of the checked locations:\n"
        f"  - {scratch_data_dir} (preferred - fast scratch drive)\n"
        f"  - {local_data_dir}\n"
        f"Please download and extract the NuInsSeg dataset to one of these locations.\n"
        f"For best performance, place it at: {local_data_dir}"
    )

def train_from_config(config_path: str):
    """
    Train StarDist model using configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup GPU
    gpu_config = config.get('gpu', {})
    if tf.config.list_physical_devices('GPU'):
        gpu_devices = tf.config.list_physical_devices('GPU')
        gpu_device = gpu_config.get('device', 0)
        if gpu_device < len(gpu_devices):
            tf.config.set_visible_devices(gpu_devices[gpu_device], 'GPU')
            gpu_memory_limit = gpu_config.get('memory_limit')
            if gpu_memory_limit:
                tf.config.experimental.set_memory_growth(gpu_devices[gpu_device], True)
                tf.config.experimental.set_memory_limit(
                    gpu_devices[gpu_device], gpu_memory_limit
                )
            print(f"\nUsing GPU {gpu_device}")
        else:
            print(f"GPU {gpu_device} not available, using CPU")
    else:
        print("No GPU available, using CPU")

    # Get data configuration and copy data if needed
    data_config = config['data']
    print("\nChecking dataset availability...")
    data_dir = copy_data_to_scratch(data_config)
    
    # Load data
    print("\nLoading NuInsSeg dataset...")
    dataset = NuInsSegDataLoader(
        data_dir, 
        data_config.get('tissues'),
        data_config.get('max_images_per_tissue')
    )
    
    if len(dataset) == 0:
        print("Error: No data loaded!")
        return
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create model configuration
    model_config = config['model']
    training_config = config['training']
    
    n_channel = 3
    
    stardist_config = Config2D(
        n_rays=model_config['n_rays'],
        grid=tuple(model_config['grid']),
        n_channel_in=n_channel,
        use_gpu=gputools_available(),
        train_epochs=training_config['epochs'],
        train_steps_per_epoch=training_config['steps_per_epoch'],
        train_batch_size=training_config['batch_size'],
        train_learning_rate=training_config['learning_rate'],
        train_patch_size=tuple(model_config['patch_size']),
        train_tensorboard=False,
    )
    
    print(f"\nModel configuration:")
    print(f"  Number of rays: {stardist_config.n_rays}")
    print(f"  Grid: {stardist_config.grid}")
    print(f"  Input channels: {stardist_config.n_channel_in}")
    print(f"  Batch size: {stardist_config.train_batch_size}")
    print(f"  Learning rate: {stardist_config.train_learning_rate}")
    print(f"  Patch size: {stardist_config.train_patch_size}\n")
    
    # Generate timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_ts = f"{model_config['model_name']}-{timestamp}"
    
    # Initialize trainer
    trainer = StarDistTrainer(
        stardist_config, 
        model_name_ts,
        model_config['model_dir']
    )
    
    # Prepare data with 3-way split
    split_config = data_config.get('splits', {'train': 0.7, 'val': 0.15, 'test': 0.15})
    X_train, Y_train, X_val, Y_val, X_test, Y_test = trainer.prepare_data(
        dataset, 
        train_split=split_config['train'],
        val_split=split_config['val'],
        test_split=split_config['test']
    )
    
    # Get wandb configuration
    wandb_config = config.get('wandb', {})

    wandb_run_name = wandb_config.get('run_name')
    if wandb_run_name:
        wandb_run_name_ts = f"{wandb_run_name}-{timestamp}"
    else:
        wandb_run_name_ts = model_name_ts
    
    # Train model
    trainer.train(
        X_train, Y_train, X_val, Y_val,
        epochs=training_config.get('epochs', 200),
        steps_per_epoch=training_config.get('steps_per_epoch'),
        use_wandb=wandb_config.get('enabled', True),
        wandb_project=wandb_config.get('project', "stardist-nuinsseg"),
        wandb_entity=wandb_config.get('entity'),
        wandb_run_name=wandb_run_name_ts,
        wandb_tags=wandb_config.get('tags'),
        log_pred_every_n_epochs=wandb_config.get('log_pred_every_n_epochs', 20),
        num_val_samples_to_log=wandb_config.get('num_val_samples_to_log', 4),
        early_stopping=True,
    )
    
    # Optimize thresholds on validation data
    trainer.optimize_thresholds(X_val, Y_val)
    
    if wandb_config.get('enabled', True):
        num_test_samples_to_log = wandb_config.get('num_test_samples_to_log', 4)
        trainer.log_test_predictions(X_test, Y_test, num_test_samples_to_log=num_test_samples_to_log)

    # Save model
    trainer.save_model()

    # Evaluate model on validation data
    print("\nEvaluating model on validation data...")
    trainer.evaluate(X_val, Y_val, data_type="validation")
    
    # Evaluate model on test data (final evaluation)
    print("Evaluating model on test data...")
    trainer.evaluate(X_test, Y_test, data_type="test")
    
    if wandb_config.get('enabled', True) and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("wandb run finished")
    
    print(f"\n\nTraining completed! Model saved in: {model_config['model_dir']}/{model_name_ts}")
    print(f"Final test set size: {len(X_test)} samples")

def main():
    """Main entry point - expects config.yaml path as first argument or looks for config.yaml in current directory"""
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    elif len(sys.argv) == 1:
        # Look for config.yaml in configs directory
        config_path = "configs/config.yaml"
        if not os.path.exists(config_path):
            print("Usage: python train_stardist_nuinsseg.py [config.yaml]")
            print("Or place config.yaml in the configs directory")
            sys.exit(1)
    else:
        print("Usage: python train_stardist_nuinsseg.py [config.yaml]")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found!")
        sys.exit(1)
    
    train_from_config(config_path)

if __name__ == "__main__":
    main()
    # train_from_config(Path(__file__).parent / "configs/test_configs/config_test.yaml")
