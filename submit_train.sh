#!/bin/bash
#SBATCH --job-name=ddpm_train    # Job name
#SBATCH --partition=gpu_short
#SBATCH --nodes=1                # Run on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=2        # Number of CPU cores per task (adjust based on num_workers)
#SBATCH --mem=32gb               # Job memory request (adjust as needed)
##SBATCH --time=4:00:00          # Time limit hrs:min:sec (adjust based on expected runtime)
#SBATCH --gres=gpu:1             # Request 1 GPU (adjust type if needed, e.g., gpu:rtx3090:1)
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err




echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on host: $(hostname)"
echo "Allocate Gpu: $CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------------"

# Load necessary modules (adjust based on your cluster's environment management)
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 && \
module load SciPy-bundle/2023.07-gfbf-2023a && \
module load matplotlib/3.7.2-gfbf-2023a && \

# Activate your Python environment (replace 'my_env' with your environment name)
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Torchvision version: $(python -c 'import torchvision; print(torchvision.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Define dataset path (replace with the actual path on your cluster)
DATA_DIR="/well/nichols-nvs/users/peo100/diffusion/Diffusion-Models-pytorch"


# Run the Python script with arguments
# Adjust arguments as needed
python train_conditional_cluster.py \
    --run_name="DDPM_Conditional_Run1" \
    --dataset_path="$DATA_DIR/cifar10-32" \
    --epochs=100 \
    --batch_size=128 \
    --image_size=32 \
    --num_classes=10 \
    --lr=3e-4 \
    --num_workers=2 \
    --save_interval=5 \
    --models_dir="$DATA_DIR/cluster_models" \
    --results_dir="$DATA_DIR/cluster_results" \
    --runs_dir="$DATA_DIR/cluster_runs" \
    --use_ema \
    # --resume_ckpt="./cluster_models/DDPM_Conditional_Run1/ckpt_epoch_50.pt" # Example resume

echo "------------------------------------------------------------"
echo "Job finished with exit code $?"
echo "------------------------------------------------------------"
