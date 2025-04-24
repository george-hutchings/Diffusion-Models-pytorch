# train_conditional_cluster.py
import os
import sys
import argparse
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt # Keep for saving context if needed, but don't show
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# --- Assumed Imports from local files ---
# Ensure modules.py and utils.py are in the same directory or PYTHONPATH
try:
    from modules import UNet_conditional, EMA
    # If setup_logging is in utils.py, import it. Otherwise, use the one defined below.
    # from utils import setup_logging as setup_logging_external
except ImportError as e:
    print(f"Error importing local modules (modules.py/utils.py): {e}", file=sys.stderr)
    print("Please ensure modules.py and utils.py are in the same directory or accessible in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

# --- Basic Logging Configuration ---
# Logs will go to stderr/stdout, managed by the Slurm output file
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S") # Use standard date format


# --- Utility Functions ---
# (Adapted from Colab/previous versions, ensure consistency with your needs)

def setup_logging_local(run_name, models_dir="models", results_dir="results", runs_dir="runs"):
    """
    Creates necessary directories for saving models, results, and TensorBoard runs.
    Uses relative paths by default, configurable via arguments.
    """
    models_path = os.path.join(models_dir, run_name)
    results_path = os.path.join(results_dir, run_name)
    runs_path = os.path.join(runs_dir, run_name) # For TensorBoard

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(runs_path, exist_ok=True)
    logging.info(f"Created/ Ensured directories exist:")
    logging.info(f"  Models: {models_path}")
    logging.info(f"  Results: {results_path}")
    logging.info(f"  TB Runs: {runs_path}")
    return models_path, results_path, runs_path

# Optional: Define plot_images if needed for specific debugging, but plt.show() won't work
# def plot_images(images):
#     plt.figure(figsize=(16, 16))
#     plt.imshow(torch.cat([
#         torch.cat([i for i in images.cpu()], dim=-1),
#     ], dim=-2).permute(1, 2, 0).cpu())
#     plt.axis('off')
#     # plt.show() # Cannot show in non-interactive cluster job

def save_images(images, path, **kwargs):
    """Saves a batch of images as a grid to a file using torchvision and PIL."""
    try:
        # Ensure images are on CPU and in range [0, 1] for make_grid
        if images.dtype == torch.uint8:
            images_float = images.float().cpu() / 255.0
        else:
            # Assuming range [-1, 1], rescale to [0, 1]
            images_float = ((images.clamp(-1, 1) + 1) / 2).cpu()

        grid = torchvision.utils.make_grid(images_float, **kwargs)
        ndarr = grid.permute(1, 2, 0).numpy()
        im = Image.fromarray((ndarr * 255).astype(np.uint8))
        im.save(path)
        # logging.info(f"Saved image grid to: {path}") # Logged in train loop
    except Exception as e:
        logging.error(f"Failed to save image to {path}: {e}")



def get_data(args):
    """
    Sets up data transformations, downloads/loads CIFAR10 dataset, returns DataLoader.
    Downloads CIFAR10 to the path specified by args.dataset_path (or a default).
    """
    # Define transformations (same as before, adjust if needed for CIFAR10's native 32x32 size)
    transforms = torchvision.transforms.Compose([
        # CIFAR10 is 32x32. Resize if args.image_size is different.
        # Example: Resize to 64x64 if args.image_size is 64
        torchvision.transforms.Resize(args.image_size),
        # Add other augmentations if desired
        # torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)), # Might not be ideal if resizing significantly
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(), # Converts to [0, 1] tensor
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes to [-1, 1]
    ])

    # Define the root directory for the dataset download
    # Use args.dataset_path if provided, otherwise default to './data'
    dataset_root = args.dataset_path if hasattr(args, 'dataset_path') and args.dataset_path else './data'
    os.makedirs(dataset_root, exist_ok=True) # Ensure the directory exists

    logging.info(f"Loading/Downloading CIFAR10 dataset to: {dataset_root}")
    try:
        # Load the CIFAR10 training dataset
        dataset = torchvision.datasets.CIFAR10(
            root=dataset_root,
            train=True,          # Load the training set
            download=True,       # Download if not already present
            transform=transforms # Apply defined transformations
        )
        logging.info(f"CIFAR10 dataset loaded successfully with {len(dataset)} training samples.")

        # CIFAR10 always has 10 classes. Verify or set args.num_classes.
        num_found_classes = 10
        if hasattr(args, 'num_classes'):
            if args.num_classes is None:
                args.num_classes = num_found_classes
                logging.info(f"Set number of classes to {args.num_classes} (CIFAR10 default)")
            elif args.num_classes != num_found_classes:
                 logging.warning(f"Provided num_classes ({args.num_classes}) does not match CIFAR10 classes ({num_found_classes}). Using {num_found_classes}.")
                 args.num_classes = num_found_classes
        else:
            # If args object doesn't even have num_classes attribute, create it
            logging.warning("args object missing 'num_classes'. Setting to 10 for CIFAR10.")
            args.num_classes = num_found_classes


    except Exception as e:
        logging.error(f"Error loading/downloading CIFAR10 dataset: {e}")
        sys.exit(1)

    # Ensure batch_size and num_workers are available in args, provide defaults if not
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 4

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True # Speeds up CPU->GPU transfer if using GPU
    )
    logging.info(f"DataLoader created with batch size {batch_size}, {num_workers} workers.")
    return dataloader



# --- Diffusion Process Implementation ---
class Diffusion:
    """Handles diffusion math: noise schedule, forward/backward process with CFG."""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t.to(self.alpha_hat.device)])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t.to(self.alpha_hat.device)])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat.to(x.device) * x + sqrt_one_minus_alpha_hat.to(x.device) * Ɛ, Ɛ

    def sample_timesteps(self, n):
        # Returns CPU tensor, move to device in training loop
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        """Generates images conditioned on labels using CFG."""
        # logging.info(f"Sampling {n} new images....") # Logged in train loop
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            if labels is not None:
                 labels = labels.to(self.device) # Ensure labels are on device

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Sampling", leave=False, file=sys.stdout): # Use sys.stdout for tqdm in Slurm
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels) # Pass labels (can be None)

                if cfg_scale > 0 and labels is not None: # Apply CFG only if labels are provided and scale > 0
                    uncond_predicted_noise = model(x, t, None) # Unconditional prediction
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                term1 = 1 / torch.sqrt(alpha)
                term2 = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
                term3 = torch.sqrt(beta) * noise
                x = term1 * (x - term2 * predicted_noise) + term3
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2 # Rescale [-1, 1] -> [0, 1]
        x = (x * 255).type(torch.uint8)
        return x.cpu() # Return results on CPU


# --- Training Function ---
def train(args):
    """Main training loop"""
    # Setup output directories
    models_path, results_path, runs_path = setup_logging_local(args.run_name, args.models_dir, args.results_dir, args.runs_dir)

    # Set device
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    args.device = device # Store device in args for easy access

    # Get data loader
    dataloader = get_data(args) # num_classes might be updated here

    # Initialize model
    try:
        model = UNet_conditional(num_classes=args.num_classes, device=device, c_in=args.c_in, time_dim=args.time_dim).to(device)
    except TypeError: # If model signature doesn't include device, c_in, time_dim
        logging.warning("UNet_conditional might not accept device/c_in/time_dim args in constructor. Initializing without them.")
        model = UNet_conditional(num_classes=args.num_classes).to(device)

    logging.info(f"Model: {type(model).__name__}, Num classes: {args.num_classes}")
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {num_params / 1e6:.2f} M")

    # Initialize optimizer, loss, diffusion helper, TensorBoard logger
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.image_size, device=device, beta_start=args.beta_start, beta_end=args.beta_end)
    logger = SummaryWriter(runs_path)
    l = len(dataloader)
    logging.info(f"Dataloader length: {l} batches")

    # Initialize EMA
    if args.use_ema:
        ema = EMA(args.ema_beta)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)
        logging.info(f"Using EMA with beta={args.ema_beta}")
    else:
        ema = None
        ema_model = None

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_ckpt:
        ckpt_path = args.resume_ckpt
        if not os.path.exists(ckpt_path):
             logging.warning(f"Resume checkpoint not found: {ckpt_path}. Starting from scratch.")
        else:
            try:
                logging.info(f"Loading checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1 # Start from next epoch

                if args.use_ema and 'ema_model_state_dict' in checkpoint:
                     ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                     # Optionally load EMA step state if saved in EMA class
                     if hasattr(ema, 'load_state_dict') and 'ema_state_dict' in checkpoint:
                         ema.load_state_dict(checkpoint['ema_state_dict'])
                elif args.use_ema:
                    logging.warning("EMA enabled but no EMA state found in checkpoint. Re-initializing EMA model from loaded model.")
                    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

                logging.info(f"Resuming training from epoch {start_epoch}")

            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
                start_epoch = 0 # Reset epoch if loading failed


    # --- Training Loop ---
    logging.info(f"Starting training from epoch {start_epoch+1} to {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        logging.info(f"--- Starting Epoch {epoch_num}/{args.epochs} ---")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_num}/{args.epochs}", file=sys.stdout, dynamic_ncols=True)
        epoch_loss = 0.0
        model.train() # Ensure model is in training mode

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            # Classifier-Free Guidance Training (randomly drop labels)
            if np.random.random() < args.cfg_drop_prob:
                labels_for_model = None
            else:
                labels_for_model = labels

            predicted_noise = model(x_t, t, labels_for_model)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema:
                ema.step_ema(ema_model, model)

            current_loss = loss.item()
            epoch_loss += current_loss
            pbar.set_postfix(MSE=current_loss)
            logger.add_scalar("Loss/Step", current_loss, global_step=epoch * l + i)

        avg_epoch_loss = epoch_loss / l
        logger.add_scalar("Loss/Epoch", avg_epoch_loss, global_step=epoch_num)
        logging.info(f"Epoch {epoch_num} Average Loss: {avg_epoch_loss:.4f}")

        # --- Periodic Sampling and Saving ---
        if epoch_num % args.save_interval == 0 or epoch == args.epochs - 1:
            logging.info(f"Epoch {epoch_num}: Sampling images and saving checkpoints...")
            model.eval() # Switch to eval mode for sampling

            # Use fixed labels for consistent sampling across epochs
            num_sampling_classes = min(args.num_classes, 10) # Sample max 10 classes or num_classes
            sample_labels = torch.arange(num_sampling_classes).long().to(device)

            # Sample images using the standard model
            logging.info("Sampling from standard model...")
            sampled_images = diffusion.sample(model, n=len(sample_labels), labels=sample_labels, cfg_scale=args.cfg_scale)
            save_path = os.path.join(results_path, f"{epoch_num}.jpg")
            save_images(sampled_images, save_path, nrow=num_sampling_classes)
            logging.info(f"Saved standard model samples to {save_path}")

            # Sample images using the EMA model if enabled
            if ema_model:
                logging.info("Sampling from EMA model...")
                ema_sampled_images = diffusion.sample(ema_model, n=len(sample_labels), labels=sample_labels, cfg_scale=args.cfg_scale)
                save_path_ema = os.path.join(results_path, f"{epoch_num}_ema.jpg")
                save_images(ema_sampled_images, save_path_ema, nrow=num_sampling_classes)
                logging.info(f"Saved EMA model samples to {save_path_ema}")

            # Save checkpoints
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss, # Store epoch loss
                'args': args # Store args used for this training run
            }
            ckpt_path = os.path.join(models_path, f"ckpt_epoch_{epoch_num}.pt")
            torch.save(checkpoint, ckpt_path)
            logging.info(f"Saved checkpoint to {ckpt_path}")

            if ema_model:
                 checkpoint['ema_model_state_dict'] = ema_model.state_dict()
                 # If EMA class has state to save:
                 # if hasattr(ema, 'state_dict'):
                 #     checkpoint['ema_state_dict'] = ema.state_dict()
                 ckpt_path_ema = os.path.join(models_path, f"ema_ckpt_epoch_{epoch_num}.pt")
                 torch.save(checkpoint, ckpt_path_ema) # Save EMA state in a separate or combined checkpoint
                 logging.info(f"Saved EMA checkpoint to {ckpt_path_ema}")

            model.train() # Switch back to train mode

    logger.close()
    logging.info("--- Training Finished ---")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Conditional Diffusion Model Training Script")

    # Paths and Logging
    parser.add_argument('--run_name', type=str, default="DDPM_conditional", help="Name for the training run (used for logging/saving)")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the training dataset (ImageFolder format)")
    parser.add_argument('--models_dir', type=str, default="models", help="Base directory to save model checkpoints")
    parser.add_argument('--results_dir', type=str, default="results", help="Base directory to save sample images")
    parser.add_argument('--runs_dir', type=str, default="runs", help="Base directory for TensorBoard logs")
    parser.add_argument('--save_interval', type=int, default=10, help="Save checkpoints and samples every N epochs")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Path to checkpoint file to resume training from")

    # Model Hyperparameters
    parser.add_argument('--image_size', type=int, default=32, help="Target image size")
    parser.add_argument('--num_classes', type=int, default=None, help="Number of classes for conditioning (if None, inferred from dataset)")
    parser.add_argument('--c_in', type=int, default=3, help='Number of input channels for UNet')
    parser.add_argument('--time_dim', type=int, default=256, help='Dimension of time embedding for UNet')

    # Diffusion Hyperparameters
    parser.add_argument('--noise_steps', type=int, default=1000, help="Number of diffusion steps (T)")
    parser.add_argument('--beta_start', type=float, default=1e-4, help="Starting beta value")
    parser.add_argument('--beta_end', type=float, default=0.02, help="Ending beta value")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per device")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument('--force_cpu', action='store_true', help="Force training on CPU even if CUDA is available")

    # EMA Hyperparameters
    parser.add_argument('--use_ema', action='store_true', default=True, help="Use Exponential Moving Average") # Default to True
    parser.add_argument('--ema_beta', type=float, default=0.995, help="Beta value for EMA")

    # Classifier-Free Guidance (CFG) Hyperparameters
    parser.add_argument('--cfg_drop_prob', type=float, default=0.1, help="Probability of dropping labels during training for CFG")
    parser.add_argument('--cfg_scale', type=float, default=3.0, help="Scale for CFG during sampling")


    args = parser.parse_args()

    # Log the arguments
    logging.info("----- Configuration -----")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("-------------------------")

    # Start training
    train(args)


if __name__ == "__main__":
    main()
