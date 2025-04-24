# Import necessary libraries
import os                     # For operating system interactions (e.g., file paths)
import copy                   # For creating deep copies of objects (used for EMA model)
import numpy as np            # For numerical operations (used for random number generation)
import torch                  # PyTorch core library
import torch.nn as nn         # Neural network modules from PyTorch
from tqdm import tqdm         # For displaying progress bars
from torch import optim       # Optimization algorithms (e.g., AdamW)
from utils import * # Import utility functions (assumed: get_data, save_images, setup_logging, plot_images)
from modules import UNet_conditional, EMA # Import the conditional U-Net and EMA classes from modules.py
import logging                # For logging information
from torch.utils.tensorboard import SummaryWriter # For TensorBoard logging

# Configure basic logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# --- Diffusion Process Implementation (Conditional) ---
class Diffusion:
    """
    Handles the diffusion process mathematics for a conditional model,
    including noise scheduling, forward process (noising), and backward process
    (sampling/denoising) with classifier-free guidance.
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        """
        Initializes the Diffusion helper class.

        Args:
            noise_steps (int): Total number of timesteps T.
            beta_start (float): Starting value of beta (variance).
            beta_end (float): Ending value of beta.
            img_size (int): Spatial size (height/width) of the images.
            device (str): Device for computations ('cuda' or 'cpu').
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Define the noise schedule (betas) and precompute alphas/alpha_hats
        self.beta = self.prepare_noise_schedule().to(device)  # β_t
        self.alpha = 1. - self.beta                            # α_t = 1 - β_t
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)       # ᾱ_t = Π_{s=1}^t α_s

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        """
        Creates the linear variance schedule (beta values).

        Returns:
            torch.Tensor: Tensor of beta values.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Applies noise to images according to the forward diffusion process formula (q(x_t | x_0)).
        Formula: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε, where ε ~ N(0, I)

        Args:
            x (torch.Tensor): Initial clean images (x_0), shape (batch, channels, height, width).
            t (torch.Tensor): Timesteps for each image, shape (batch,).

        Returns:
            tuple:
                - torch.Tensor: Noised images (x_t).
                - torch.Tensor: Noise (ε) added.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x) # Sample standard Gaussian noise
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """
        Samples random timesteps for training.

        Args:
            n (int): Number of timesteps to sample (batch size).

        Returns:
            torch.Tensor: Tensor of n random timesteps [1, noise_steps).
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        """
        Generates new images conditioned on labels using the trained model and classifier-free guidance (CFG).

        Args:
            model (nn.Module): Trained U-Net model (conditional).
            n (int): Number of images to generate.
            labels (torch.Tensor): Conditioning labels for the images, shape (n,).
            cfg_scale (float): Scale factor for Classifier-Free Guidance.
                               0: Unconditional generation.
                               1: Purely conditional generation.
                               >1: Mix guidance towards the condition.

        Returns:
            torch.Tensor: Generated images, scaled to [0, 255], uint8.
        """
        logging.info(f"Sampling {n} new images....")
        model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            # Start with pure Gaussian noise (x_T)
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # Iterate backwards from T-1 down to 0
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Sampling"):
                # Current timestep tensor
                t = (torch.ones(n) * i).long().to(self.device)
                # Predict noise conditioned on labels
                predicted_noise = model(x, t, labels)
                # --- Classifier-Free Guidance ---
                if cfg_scale > 0:
                    # Predict noise unconditionally (by passing None as labels)
                    uncond_predicted_noise = model(x, t, None)
                    # Interpolate between unconditional and conditional predictions
                    # predicted_noise = uncond + cfg_scale * (cond - uncond)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                # Retrieve diffusion constants for timestep t
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Add noise for steps t > 1
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) # No noise added at the last step

                # DDPM sampling step using the (potentially CFG-modified) predicted noise
                # x_{t-1} = (1/√α_t) * (x_t - ((1 - α_t) / √(1 - ᾱ_t)) * ε_θ(x_t, t)) + √β_t * z
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train() # Set model back to training mode
        # Post-process the final generated image (x_0)
        x = (x.clamp(-1, 1) + 1) / 2 # Rescale [-1, 1] -> [0, 1]
        x = (x * 255).type(torch.uint8) # Rescale [0, 1] -> [0, 255] and convert to uint8
        return x


# --- Training Function (Conditional with EMA and CFG) ---
def train(args):
    """
    Sets up and runs the training loop for the conditional diffusion model
    with EMA and classifier-free guidance training.

    Args:
        args (argparse.Namespace): Object containing training configuration arguments.
    """
    # Setup logging directory (function assumed from utils)
    setup_logging(args.run_name)
    device = args.device
    # Get data loader (function assumed from utils)
    dataloader = get_data(args)
    # Initialize the *conditional* U-Net model
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    # Initialize AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Define MSE loss
    mse = nn.MSELoss()
    # Initialize Diffusion helper
    diffusion = Diffusion(img_size=args.image_size, device=device)
    # Initialize TensorBoard logger
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    # Get number of batches for logging steps
    l = len(dataloader)
    # Initialize EMA helper with decay factor 0.995
    ema = EMA(0.995)
    # Create a deep copy of the model for EMA, set to eval mode, and disable gradients
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        # Loop through batches (images and labels)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            # Sample timesteps
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # Create noised images and get the noise
            x_t, noise = diffusion.noise_images(images, t)

            # --- Classifier-Free Guidance Training ---
            # With a 10% probability, drop the labels (set to None)
            # This forces the model to learn unconditional generation as well.
            if np.random.random() < 0.1:
                labels = None

            # Get the model's noise prediction (conditional or unconditional)
            predicted_noise = model(x_t, t, labels)
            # Calculate MSE loss
            loss = mse(noise, predicted_noise)

            # --- Optimization and EMA Update ---
            optimizer.zero_grad() # Reset gradients
            loss.backward()       # Compute gradients
            optimizer.step()      # Update model parameters using optimizer
            ema.step_ema(ema_model, model) # Update the EMA model weights

            # Log loss to progress bar and TensorBoard
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # --- Periodic Sampling and Saving (every 10 epochs) ---
        if epoch % 10 == 0:
            # Define labels to sample (e.g., one image for each class if num_classes=10)
            labels = torch.arange(args.num_classes).long().to(device)
            # Sample images using the standard model
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            # Sample images using the EMA model
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

            # Plot and save images (plot_images assumed from utils)
            plot_images(sampled_images) # Display images from standard model
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))

            # Save model checkpoints
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            # Save optimizer state (useful for resuming training)
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
            logging.info(f"Epoch {epoch} completed. Samples saved and checkpoints created.")


# --- Launch Function ---
def launch():
    """
    Parses arguments (or sets them manually) and starts conditional training.
    """
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args() # No command-line args defined here

    # --- Manually set training configuration ---
    args.run_name = "DDPM_conditional" # Name for the run
    args.epochs = 300                  # Number of epochs
    args.batch_size = 14               # Batch size
    args.image_size = 64               # Image size
    args.num_classes = 10              # Number of classes for conditioning (e.g., CIFAR-10)
    args.dataset_path = r"/home/hutchings/OneDrive/Documents/academic/24-25/diffusion/Diffusion-Models-pytorch/datasets/cifar10" # Path to dataset
    args.device = "cuda"               # Device
    args.lr = 3e-4                     # Learning rate

    # Start training
    train(args)


# --- Main Execution Guard ---
if __name__ == '__main__':
    # Start the training process
    launch()

    # --- Commented-out code for Sampling from a Trained Conditional Model ---
    # device = "cuda"
    # # Load the conditional U-Net architecture
    # model = UNet_conditional(num_classes=10).to(device)
    # # Load saved weights
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8 # Number of images to sample
    # # Create labels for the desired class (e.g., class 6)
    # y = torch.Tensor([6] * n).long().to(device)
    # # Sample images using the loaded model, specific labels 'y', and cfg_scale
    # # cfg_scale=0 means unconditional sampling (ignores 'y')
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # # Plot the generated images (function assumed from utils)
    # plot_images(x)
