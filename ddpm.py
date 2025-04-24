# Import necessary libraries
import os                     # For interacting with the operating system (e.g., file paths)
import torch                  # PyTorch core library
import torch.nn as nn         # Neural network modules from PyTorch
from matplotlib import pyplot as plt # For plotting images (used in commented-out section)
from tqdm import tqdm         # For displaying progress bars during loops
from torch import optim       # Optimization algorithms (e.g., AdamW)
from utils import * # Import utility functions (assumed to contain get_data, save_images, setup_logging)
from modules import UNet      # Import the U-Net model architecture defined in modules.py
import logging                # For logging information during training
from torch.utils.tensorboard import SummaryWriter # For logging metrics to TensorBoard

# Configure basic logging settings
# Format: Time - Log Level: Message
# Level: INFO means messages with level INFO, WARNING, ERROR, CRITICAL will be shown.
# Date Format: Hour:Minute:Second
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# --- Diffusion Process Implementation ---
class Diffusion:
    """
    Handles the diffusion process mathematics: noise scheduling, forward process (noising),
    and backward process (sampling/denoising).
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        """
        Initializes the Diffusion helper class.

        Args:
            noise_steps (int): The total number of timesteps T in the diffusion process.
            beta_start (float): The starting value of beta (variance) for the noise schedule.
            beta_end (float): The ending value of beta for the noise schedule.
            img_size (int): The spatial size (height/width) of the images.
            device (str): The device to run computations on ('cuda' or 'cpu').
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Define the noise schedule (betas) and precompute alphas and alpha_hats
        # These are constants derived from the schedule and used in the diffusion equations.
        self.beta = self.prepare_noise_schedule().to(device)  # β_t
        self.alpha = 1. - self.beta                            # α_t = 1 - β_t
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)       # ᾱ_t = Π_{s=1}^t α_s (cumulative product)

    def prepare_noise_schedule(self):
        """
        Creates the variance schedule (beta values) for the diffusion process.
        This implementation uses a linear schedule.

        Returns:
            torch.Tensor: A tensor of beta values from beta_start to beta_end.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) # Linear interpolation

    def noise_images(self, x, t):
        """
        Applies noise to images according to the forward diffusion process formula (q(x_t | x_0)).
        Formula: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε, where ε ~ N(0, I)

        Args:
            x (torch.Tensor): The initial clean images (x_0), shape (batch, channels, height, width).
            t (torch.Tensor): A tensor of timesteps for each image in the batch, shape (batch,).

        Returns:
            tuple:
                - torch.Tensor: The noised images (x_t) at the specified timesteps.
                - torch.Tensor: The noise (ε) that was added to each image.
        """
        # Get sqrt(ᾱ_t) for the sampled timesteps t
        # Unsqueeze to match the image tensor dimensions (batch, 1, 1, 1) for broadcasting
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        # Get sqrt(1 - ᾱ_t) for the sampled timesteps t
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        # Sample standard Gaussian noise (ε) with the same shape as x
        Ɛ = torch.randn_like(x)
        # Apply the forward process formula
        noised_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ
        return noised_image, Ɛ # Return the noised image and the noise applied

    def sample_timesteps(self, n):
        """
        Samples random timesteps for training.

        Args:
            n (int): The number of timesteps to sample (usually the batch size).

        Returns:
            torch.Tensor: A tensor of n random timesteps (integers) between 1 and noise_steps (inclusive).
        """
        # Sample n integers uniformly from the range [1, noise_steps)
        # Note: Original DDPM paper samples from [1, T], this samples [1, T-1] implicitly? Check randint docs.
        # torch.randint high is exclusive, so it samples from [1, noise_steps-1].
        # This might be intentional or a slight deviation. Often sampling t=0 (step 1 in 1-based index) isn't needed.
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """
        Generates new images using the trained model by reversing the diffusion process (sampling).
        Starts from pure noise (x_T) and iteratively denoises it using the model's predictions.

        Args:
            model (nn.Module): The trained U-Net model used to predict noise.
            n (int): The number of images to generate.

        Returns:
            torch.Tensor: The generated images, scaled to [0, 255] and converted to uint8.
        """
        logging.info(f"Sampling {n} new images....")
        model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates)
        with torch.no_grad(): # Disable gradient calculations for efficiency
            # Start with pure Gaussian noise (x_T)
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            # Iterate backwards from T-1 down to 0 (using 1 to noise_steps-1 range here)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Sampling"):
                # Create a tensor of the current timestep 'i' for all images in the batch
                t = (torch.ones(n) * i).long().to(self.device)
                # Predict the noise (ε_θ) added at timestep 'i' using the U-Net model
                predicted_noise = model(x, t)

                # Retrieve precomputed alpha, alpha_hat, and beta for the current timestep 'i'
                # Unsqueeze to match image dimensions for broadcasting
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Add noise for steps i > 1. For the last step (i=1 -> t=0), no noise is added (z=0).
                if i > 1:
                    noise = torch.randn_like(x) # Sample Gaussian noise z ~ N(0, I)
                else:
                    noise = torch.zeros_like(x) # z = 0 for the last step

                # Apply the DDPM sampling formula to get x_{t-1} from x_t:
                # x_{t-1} = (1/√α_t) * (x_t - (β_t / √(1 - ᾱ_t)) * ε_θ(x_t, t)) + √β_t * z
                # This can be rewritten as: (1/√α_t) * (x_t - ((1 - α_t) / √(1 - ᾱ_t)) * ε_θ(x_t, t)) + √β_t * z
                term1 = 1 / torch.sqrt(alpha)
                term2 = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
                term3 = torch.sqrt(beta) * noise

                x = term1 * (x - term2 * predicted_noise) + term3

        model.train() # Set the model back to training mode

        # Post-process the final generated image (x_0)
        # Clamp values to the [-1, 1] range (typical output range for models using tanh)
        x = (x.clamp(-1, 1) + 1) / 2 # Rescale from [-1, 1] to [0, 1]
        x = (x * 255).type(torch.uint8) # Rescale from [0, 1] to [0, 255] and convert to byte tensor
        return x


# --- Training Function ---
def train(args):
    """
    Sets up and runs the training loop for the diffusion model.

    Args:
        args (argparse.Namespace): An object containing training configuration arguments.
    """
    # Setup logging directory based on the run name (function assumed from utils)
    setup_logging(args.run_name)
    device = args.device
    # Get the data loader (function assumed from utils)
    dataloader = get_data(args)
    # Initialize the U-Net model and move it to the specified device
    model = UNet().to(device)
    # Initialize the AdamW optimizer with the model parameters and learning rate
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Define the loss function: Mean Squared Error (MSE) between added noise and predicted noise
    mse = nn.MSELoss()
    # Initialize the Diffusion helper class with parameters from args
    diffusion = Diffusion(img_size=args.image_size, device=device)
    # Initialize TensorBoard SummaryWriter to log metrics in the 'runs/{run_name}' directory
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    # Get the number of batches in the dataloader for logging steps
    l = len(dataloader)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        # Wrap the dataloader with tqdm for a progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, (images, _) in enumerate(pbar): # Loop through batches (images and optional labels _)
            # Move images to the target device
            images = images.to(device)
            # Sample random timesteps for the current batch
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # Generate noised images (x_t) and the corresponding noise (ε) using the forward process
            x_t, noise = diffusion.noise_images(images, t)
            # Get the model's prediction of the noise based on the noised image and timestep
            predicted_noise = model(x_t, t)
            # Calculate the MSE loss between the actual noise and the predicted noise
            loss = mse(noise, predicted_noise)

            # --- Backpropagation and Optimization ---
            optimizer.zero_grad() # Reset gradients
            loss.backward()       # Compute gradients
            optimizer.step()      # Update model parameters

            # Update the progress bar postfix with the current loss
            pbar.set_postfix(MSE=loss.item())
            # Log the loss value to TensorBoard
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # --- After each epoch ---
        # Generate sample images using the current model state
        sampled_images = diffusion.sample(model, n=images.shape[0]) # Sample a batch of images
        # Save the generated images (function assumed from utils)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        # Save the model's state dictionary (weights)
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        logging.info(f"Epoch {epoch} completed. Sampled images saved and model checkpoint saved.")


# --- Launch Function ---
def launch():
    """
    Parses command-line arguments (or sets them manually here) and starts the training process.
    """
    import argparse
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    # Parse arguments (in this case, no arguments are defined to be parsed from command line)
    args = parser.parse_args()

    # --- Manually set training configuration arguments ---
    args.run_name = "DDPM_Uncondtional" # Name for the training run (used for logging/saving)
    args.epochs = 500                   # Number of training epochs
    args.batch_size = 12                # Number of images per batch
    args.image_size = 64                # Target image size (resize happens in get_data)
    args.dataset_path = r"/home/hutchings/OneDrive/Documents/academic/24-25/diffusion/Diffusion-Models-pytorch/datasets/landscape" # Path to the dataset
    args.device = "cuda"                # Device to use ("cuda" or "cpu")
    args.lr = 3e-4                      # Learning rate for the optimizer

    # Start the training process with the configured arguments
    train(args)


# --- Main Execution Guard ---
if __name__ == '__main__':
    # Call the launch function to configure and start training
    launch()

    # --- Commented-out code for Sampling and Visualization ---
    # This section shows how to load a trained model and generate samples.
    # device = "cuda"
    # model = UNet().to(device)
    # # Load pre-trained model weights
    # ckpt = torch.load("./working/orig/ckpt.pt") # Path to the saved checkpoint
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device) # Initialize Diffusion helper
    # # Sample 8 images
    # x = diffusion.sample(model, 8)
    # print(x.shape) # Print the shape of the generated tensor
    # # Prepare images for plotting (concatenate them into a grid)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1), # Concatenate images horizontally
    # ], dim=-2).permute(1, 2, 0).cpu()) # Concatenate rows vertically (if multiple rows) and permute to (H, W, C)
    # plt.show() # Display the plot
