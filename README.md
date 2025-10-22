# StarDist NuInsSeg Training

This repository provides a training script for 2D StarDist models using the NuInsSeg dataset, configured via YAML files with automatic dataset management.

Find the Paper [here](https://www.nature.com/articles/s41597-024-03117-2).
Download the dataset [here](https://zenodo.org/records/10518968/files/NuInsSeg.zip?download=1).


### Basic Usage

1. Configure your training parameters in `config.yaml`
2. Run the training:
   ```bash
   # With explicit config file path
   python train_stardist_nuinsseg.py configs/config.yaml
   
   # Or if config.yaml is in configuration directory "configs/"
   python train_stardist_nuinsseg.py
   ```


## Dataset Management

The script automatically handles dataset location and optional copying:

- `data_dir`: Primary location where dataset should be stored
- `scratch_dir`: Alternative location to check for dataset (default: "/scratch")  
- `copy_to_scratch`: Whether to automatically copy dataset from scratch to local storage (default: true)


## Configuration

Edit configuration files (`config_template.yaml`) to customize your training. The system now supports **Fixed Test Sets Strategy** for comparative studies:


### Fixed Test Sets Strategy (Recommended for Model Comparison)

```yaml
data:
  data_dir: "/path/to/data"
  scratch_data_dir: "/path/to/data"
  copy_to_scratch: true
  tissues: mouse*  # null, string or list of strings (accepts mix of glob patterns and specific names)
  max_images_per_tissue: null
  use_fixed_test_sets: true  # use fixed human_test & mouse_test sets
  
splits:
  train: 0.8
  val: 0.2
  # test: 0.2 # only used when use_fixed_test_sets: false (legacy mode)

model:
  model_name: "stardist-nuinsseg"
  model_dir: "models"
  n_rays: 32
  grid: [2, 2]
  patch_size: [512, 512] # [256, 256]

training:
  epochs: 400
  steps_per_epoch: null  # if null calculated as dataset_size / batch_size
  batch_size: 4
  learning_rate: 0.0003

wandb:
  enabled: true
  project: "stardist-nuinsseg"
  entity: null
  run_name: "nuinsseg-mouse" # automatically timestamped
  tags: ["human-mouse"]
  log_pred_every_n_epochs: 20
  num_val_samples_to_log: 4 # number of validation samples to log to wandb media section
  num_test_samples_to_log: 4 # number of test samples to log to wandb media section

gpu:
  device: 5
  memory_limit: null # memory limit (MB)
```

### How Fixed Test Sets Work

1. **Fixed Test Set Creation**: 
   - Creates `human_test`: ~1% of images from each human tissue folder (min 1 image)
   - Creates `mouse_test`: ~1% of images from each mouse tissue folder (min 1 image)
   - These test sets are **consistent across all training runs**

2. **Training Data Selection**:
   - Removes fixed test images from available pool
   - Applies `tissues` filter to remaining data for train/val split
   - Enables fair comparison between models trained on different tissue subsets

3. **Evaluation**:
   - All models evaluated on same `human_test` and `mouse_test` sets
   - Separate metrics reported for human, mouse, and combined performance


### Example Training Scenarios

```bash
# Train on all tissues, test on both human and mouse
python train_stardist_nuinsseg.py configs/config_all_tissues.yaml

# Train on human tissues only, test on both human and mouse  
python train_stardist_nuinsseg.py configs/config_human_only.yaml

# Train on mouse tissues only, test on both human and mouse
python train_stardist_nuinsseg.py configs/config_mouse_only.yaml
```

### Legacy Mode (Traditional Random Split)

For backward compatibility, set `use_fixed_test_sets: false`:

```yaml
data:
  use_fixed_test_sets: false
splits:
  train: 0.7
  val: 0.15
  test: 0.15  # Required when use_fixed_test_sets: false
```

## Features

- **Fixed Test Sets Strategy**: Consistent human/mouse test sets for model comparison
- **YAML Configuration**: Training parameters configured via YAML files
- **Automatic Dataset Management**: Finds and copies dataset from scratch drives
- **Data Loading**: Automatic loading and preprocessing of NuInsSeg dataset
- **Data Augmentation**: Built-in augmentation with flips, rotations, and intensity changes
- **Weights & Biases Integration**: Optional experiment tracking and metric logging
- **Checkpointing**: Automatic model checkpointing during training
- **Early Stopping**: Prevents overfitting with early stopping
- **Threshold Optimization**: Automatic optimization of thresholds for NMS postprocessing
- **Comparative Evaluation**: Separate metrics for human, mouse, and combined test performance
