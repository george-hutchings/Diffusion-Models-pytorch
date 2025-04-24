# Import necessary libraries
import os                     # For interacting with the operating system (creating directories)
import torch                  # PyTorch core library
import torchvision            # PyTorch library for computer vision tasks (datasets, transforms, utils)
from PIL import Image         # Python Imaging Library for image manipulation (saving images)
from matplotlib import pyplot as plt # Library for creating plots and visualizations
from torch.utils.data import DataLoader # PyTorch utility for loading data in batches


# --- Image Plotting Function ---
def plot_images(images):
    """
    Displays a batch of images in a grid using Matplotlib.

    Args:
        images (torch.Tensor): A tensor containing a batch of images,
                               typically with shape (batch_size, channels, height, width)
                               and values in the range [0, 1] or [-1, 1] (will be visualized).
                               Assumes RGB channel order if channels=3.
    """
    # Create a large figure to display the images
    plt.figure(figsize=(32, 32))

    # Process the image tensor for display:
    # 1. images.cpu(): Move the tensor to the CPU if it's on the GPU.
    # 2. [i for i in images.cpu()]: Create a list of individual image tensors from the batch.
    # 3. torch.cat([...], dim=-1): Concatenate images horizontally along the width dimension.
    #    Example: If batch size is 4, creates one row [img1, img2, img3, img4].
    # 4. torch.cat([...], dim=-2): Concatenate rows vertically (if needed, here only one row).
    #    Result is a single large tensor representing the grid.
    # 5. .permute(1, 2, 0): Change tensor dimensions from (C, H, W) to (H, W, C)
    #    which is the format expected by Matplotlib's imshow.
    # 6. .cpu(): Ensure the final tensor is on the CPU.
    image_grid = torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1), # Concatenate images horizontally
    ], dim=-2).permute(1, 2, 0).cpu() # Concatenate vertically (if multiple rows) and permute C,H,W -> H,W,C

    # Display the image grid
    plt.imshow(image_grid)
    # Show the plot window
    plt.show()


# --- Image Saving Function ---
def save_images(images, path, **kwargs):
    """
    Saves a batch of images as a grid to a file using torchvision and PIL.

    Args:
        images (torch.Tensor): A tensor containing a batch of images,
                               typically shape (batch_size, channels, height, width).
                               Expected value range often [0, 1] or [0, 255] uint8 for direct saving.
        path (str): The file path where the image grid should be saved.
        **kwargs: Additional keyword arguments to pass to torchvision.utils.make_grid
                  (e.g., nrow for number of images per row, padding).
    """
    # Create a grid tensor from the batch of images.
    # make_grid arranges the images into a grid format (single tensor).
    grid = torchvision.utils.make_grid(images, **kwargs)

    # Convert the grid tensor to a NumPy array suitable for saving:
    # 1. .permute(1, 2, 0): Change tensor dimensions from (C, H, W) to (H, W, C).
    # 2. .to('cpu'): Ensure the tensor is on the CPU.
    # 3. .numpy(): Convert the tensor to a NumPy array.
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()

    # Create a PIL Image object from the NumPy array.
    im = Image.fromarray(ndarr)
    # Save the PIL image to the specified path.
    im.save(path)


# --- Data Loading Function ---
def get_data(args):
    """
    Sets up data transformations, loads an image dataset from a folder,
    and returns a DataLoader.

    Args:
        args (argparse.Namespace or similar object): An object containing configuration
                                                     parameters like image_size, dataset_path,
                                                     and batch_size.

    Returns:
        torch.utils.data.DataLoader: The DataLoader object for iterating over the dataset.
    """
    # Define a sequence of image transformations to apply to each image.
    transforms = torchvision.transforms.Compose([
        # Resize images to a slightly larger size first.
        # Example: If target size is 64, resize to 80 (64 + 1/4*64).
        torchvision.transforms.Resize(80),
        # Apply random resized cropping:
        # - Crop a random portion of the image.
        # - Resize that portion to the final args.image_size.
        # - scale=(0.8, 1.0) ensures the cropped area is between 80% and 100% of the original image.
        # This acts as data augmentation.
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        # Convert the PIL Image (or NumPy array) to a PyTorch tensor.
        # Scales pixel values from [0, 255] to [0.0, 1.0].
        torchvision.transforms.ToTensor(),
        # Normalize the tensor image:
        # Subtracts the mean (0.5) and divides by the standard deviation (0.5) for each channel.
        # This maps the pixel values from [0.0, 1.0] to [-1.0, 1.0].
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset using ImageFolder:
    # - Assumes the dataset is organized in subfolders where each subfolder represents a class.
    # - args.dataset_path points to the root directory containing these subfolders.
    # - Applies the defined transforms to each image as it's loaded.
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)

    # Create a DataLoader:
    # - Wraps the dataset to provide an iterable over batches.
    # - batch_size=args.batch_size specifies the number of samples per batch.
    # - shuffle=True randomizes the order of samples in each epoch.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Return the configured DataLoader
    return dataloader


# --- Logging Setup Function ---
def setup_logging(run_name):
    """
    Creates necessary directories for saving model checkpoints and results/samples
    based on the provided run name.

    Args:
        run_name (str): The name of the current training run, used to create
                        specific subdirectories.
    """
    # Create the base 'models' directory if it doesn't exist.
    os.makedirs("models", exist_ok=True)
    # Create the base 'results' directory if it doesn't exist.
    os.makedirs("results", exist_ok=True)
    # Create a subdirectory within 'models' specific to this run_name.
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    # Create a subdirectory within 'results' specific to this run_name.
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
