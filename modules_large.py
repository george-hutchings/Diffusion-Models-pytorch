# Import necessary libraries from PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Exponential Moving Average (EMA) ---
# EMA is often used in training deep learning models, especially generative models like diffusion models.
# It maintains a "shadow" copy of the model's weights that represents a moving average of the weights
# over time. This can lead to more stable and better-performing models at inference time.
class EMA:
    """Implements Exponential Moving Average for model parameters."""
    def __init__(self, beta):
        """
        Initializes the EMA object.
        Args:
            beta (float): The decay factor for the moving average (often close to 1.0, e.g., 0.999).
        """
        super().__init__()
        self.beta = beta # Decay factor
        self.step = 0    # Counter for the number of updates

    def update_model_average(self, ma_model, current_model):
        """
        Update the moving average model (ma_model) with the current model's parameters.
        Args:
            ma_model (nn.Module): The model whose parameters are the moving average (target).
            current_model (nn.Module): The model whose current parameters are used for the update (source).
        """
        # Iterate through the parameters of both models simultaneously
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            # Get the data tensors for the current weights and the moving average weights
            old_weight, up_weight = ma_params.data, current_params.data
            # Update the moving average weights using the EMA formula
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Calculate the updated moving average for a single parameter tensor.
        EMA formula: new_average = beta * old_average + (1 - beta) * new_value
        Args:
            old (torch.Tensor): The previous moving average value (or None if it's the first update).
            new (torch.Tensor): The new value from the current model.
        Returns:
            torch.Tensor: The updated moving average value.
        """
        if old is None:
            # If it's the first update, the average is just the new value
            return new
        # Apply the EMA update rule
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Performs one EMA update step. Can optionally delay the start of EMA.
        Args:
            ema_model (nn.Module): The moving average model.
            model (nn.Module): The current training model.
            step_start_ema (int): The training step number after which EMA updates should begin.
        """
        # Only start EMA updates after a certain number of steps
        if self.step < step_start_ema:
            # Before starting EMA, ensure the ema_model has the same weights as the current model
            self.reset_parameters(ema_model, model)
            self.step += 1
            return # Skip the update
        # Perform the EMA update
        self.update_model_average(ema_model, model)
        # Increment the step counter
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Copies the parameters from the current model to the EMA model.
        Used to initialize the EMA model or reset it before EMA updates start.
        Args:
            ema_model (nn.Module): The moving average model (destination).
            model (nn.Module): The current training model (source).
        """
        ema_model.load_state_dict(model.state_dict())


# --- Self Attention Block ---
# Implements a self-attention mechanism, often used in transformer-based architectures
# and incorporated into U-Nets for capturing long-range dependencies in the feature maps.
class SelfAttention(nn.Module):
    """A Self-Attention block with Multihead Attention and a Feed-Forward network."""
    def __init__(self, channels, size):
        """
        Initializes the SelfAttention block.
        Args:
            channels (int): Number of input and output channels.
            size (int): Height/Width of the input feature map (assumed square).
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size # Spatial dimension (height/width)
        # MultiheadAttention layer:
        # - channels: Embedding dimension for each input token.
        # - 4: Number of attention heads.
        # - batch_first=True: Expects input tensors in the format (batch, sequence, features).
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        # Layer Normalization applied before attention
        self.ln = nn.LayerNorm([channels])
        # Feed-forward network applied after attention, typical in transformer blocks
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),        # Layer Normalization
            nn.Linear(channels, channels),  # Linear layer
            nn.GELU(),                      # GELU activation function
            nn.Linear(channels, channels),  # Linear layer
        )

    def forward(self, x):
        """
        Forward pass for the SelfAttention block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Reshape input for MultiheadAttention: (batch, channels, height, width) -> (batch, channels, height*width) -> (batch, height*width, channels)
        # This treats each spatial location (pixel) as a token in a sequence.
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        # Apply Layer Normalization before attention
        x_ln = self.ln(x)
        # Apply Multihead Self-Attention. Query, Key, and Value are all the normalized input (x_ln).
        # Returns the attention output and attention weights (which are ignored here with _)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        # Add residual connection (skip connection) from the original input x
        attention_value = attention_value + x
        # Apply the feed-forward network with another residual connection
        attention_value = self.ff_self(attention_value) + attention_value
        # Reshape back to the original image format: (batch, height*width, channels) -> (batch, channels, height*width) -> (batch, channels, height, width)
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


# --- Double Convolution Block ---
# A common building block in U-Nets, consisting of two consecutive convolutional layers
# with normalization and activation functions.
class DoubleConv(nn.Module):
    """A block with two convolutional layers, optional residual connection."""
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        """
        Initializes the DoubleConv block.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of channels in the intermediate layer. Defaults to out_channels.
            residual (bool, optional): Whether to use a residual connection. Defaults to False.
        """
        super().__init__()
        self.residual = residual
        # If mid_channels is not specified, set it equal to out_channels
        if not mid_channels:
            mid_channels = out_channels
        # Define the sequence of layers
        self.double_conv = nn.Sequential(
            # First convolution: in_channels -> mid_channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # 3x3 kernel, padding=1 keeps spatial size same, bias=False because GroupNorm has affine param
            nn.GroupNorm(1, mid_channels), # Group Normalization with 1 group is equivalent to Instance Normalization
            nn.GELU(),                     # GELU activation function
            # Second convolution: mid_channels -> out_channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        ) # Note: No final activation here, it's applied after the residual connection if used.

    def forward(self, x):
        """
        Forward pass for the DoubleConv block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        if self.residual:
            # If residual connection is enabled, add input x to the output of double_conv and apply GELU
            return F.gelu(x + self.double_conv(x))
        else:
            # If no residual connection, just return the output of double_conv
            return self.double_conv(x)


# --- Downsampling Block (Encoder part of U-Net) ---
# This block reduces the spatial dimensions (height/width) and increases the channel dimension.
# It also incorporates time embedding information.
class Down(nn.Module):
    """Downscaling block with MaxPool, DoubleConv, and time embedding integration."""
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        Initializes the Down block.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_dim (int, optional): Dimension of the time embedding. Defaults to 256.
        """
        super().__init__()
        # Sequence for downsampling and feature extraction
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # Halve the spatial dimensions (height and width)
            DoubleConv(in_channels, in_channels, residual=True), # Apply DoubleConv with residual connection (maintains channels)
            DoubleConv(in_channels, out_channels),             # Apply DoubleConv to change channels from in_channels to out_channels
        )

        # Layer to process the time embedding and project it to the output channel dimension
        self.emb_layer = nn.Sequential(
            nn.SiLU(), # SiLU (Swish) activation function
            nn.Linear(
                emb_dim,      # Input dimension is the time embedding dimension
                out_channels  # Output dimension matches the output channels of the convolutions
            ),
        )

    def forward(self, x, t):
        """
        Forward pass for the Down block.
        Args:
            x (torch.Tensor): Input feature map from the previous layer.
            t (torch.Tensor): Time embedding tensor.
        Returns:
            torch.Tensor: Output feature map after downsampling and embedding addition.
        """
        # Apply max pooling and double convolutions
        x = self.maxpool_conv(x)
        # Process the time embedding:
        # 1. Pass time embedding 't' through the embedding layer (SiLU + Linear).
        # 2. Reshape the embedding: Add spatial dimensions (height, width) -> (batch, out_channels, 1, 1).
        # 3. Repeat the embedding across the spatial dimensions to match the feature map 'x'.
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # Add the processed time embedding to the feature map
        return x + emb


# --- Upsampling Block (Decoder part of U-Net) ---
# This block increases the spatial dimensions and potentially reduces the channel dimension.
# It typically involves concatenating feature maps from the corresponding encoder layer (skip connection).
class Up(nn.Module):
    """Upscaling block with Upsample, concatenation, DoubleConv, and time embedding integration."""
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        Initializes the Up block.
        Args:
            in_channels (int): Number of input channels (from the previous decoder layer + skip connection).
            out_channels (int): Number of output channels.
            emb_dim (int, optional): Dimension of the time embedding. Defaults to 256.
        """
        super().__init__()

        # Upsampling layer: Doubles the height and width using bilinear interpolation
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Convolutional layers applied after upsampling and concatenation
        # Note: The input channels to the first DoubleConv is `in_channels` because it receives
        # the concatenated tensor (skip_connection + upsampled_feature_map).
        # Usually, the skip connection has `in_channels // 2` channels and the upsampled map also has `in_channels // 2`.
        self.conv = nn.Sequential(
            # DoubleConv with residual connection. Input channels = in_channels (skip + upsampled)
            DoubleConv(in_channels, in_channels, residual=True),
            # DoubleConv to reduce channels. Input channels = in_channels, Output channels = out_channels.
            # It uses `in_channels // 2` as the intermediate channel dimension.
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # Layer to process the time embedding (same as in the Down block)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels # Project embedding to the output channel dimension
            ),
        )

    def forward(self, x, skip_x, t):
        """
        Forward pass for the Up block.
        Args:
            x (torch.Tensor): Input feature map from the previous (lower resolution) decoder layer.
            skip_x (torch.Tensor): Feature map from the corresponding encoder layer (skip connection).
            t (torch.Tensor): Time embedding tensor.
        Returns:
            torch.Tensor: Output feature map after upsampling, concatenation, and embedding addition.
        """
        # Upsample the input feature map x
        x = self.up(x)
        # Concatenate the upsampled feature map 'x' with the skip connection feature map 'skip_x' along the channel dimension (dim=1)
        # Example shapes: skip_x (B, C1, H, W), x (B, C2, H, W) -> cat (B, C1+C2, H, W)
        # Here, in_channels = C1 + C2
        x = torch.cat([skip_x, x], dim=1)
        # Apply the double convolution blocks
        x = self.conv(x)
        # Process and add the time embedding (similar to the Down block)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# --- U-Net Architecture ---
# Assembles the Down, Up, SelfAttention, and DoubleConv blocks into a U-Net model.
# This version is typically used for unconditional generation in diffusion models.
class UNet(nn.Module):
    """A U-Net model architecture with time embedding and self-attention layers."""
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        """
        Initializes the U-Net model.
        Args:
            c_in (int, optional): Number of input channels (e.g., 3 for RGB images). Defaults to 3.
            c_out (int, optional): Number of output channels (e.g., 3 for predicted noise). Defaults to 3.
            time_dim (int, optional): Dimension of the time embedding vector. Defaults to 256.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to "cuda".
        """
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # --- Encoder ---
        # Initial convolution block (maintains spatial size, increases channels)
        self.inc = DoubleConv(c_in, 64)
        # Downsampling block 1 (64 -> 128 channels, size / 2)
        self.down1 = Down(64, 128)
        # Self-attention block 1 (operates at size 32x32)
        self.sa1 = SelfAttention(128, 32)
        # Downsampling block 2 (128 -> 256 channels, size / 4)
        self.down2 = Down(128, 256)
        # Self-attention block 2 (operates at size 16x16)
        self.sa2 = SelfAttention(256, 16)
        # Downsampling block 3 (256 -> 256 channels, size / 8)
        self.down3 = Down(256, 256)
        # Self-attention block 3 (operates at size 8x8)
        self.sa3 = SelfAttention(256, 8)

        # --- Bottleneck ---
        # Three double convolution blocks at the lowest resolution (size / 8)
        self.bot1 = DoubleConv(256, 512) # 256 -> 512 channels
        self.bot2 = DoubleConv(512, 512) # 512 -> 512 channels
        self.bot3 = DoubleConv(512, 256) # 512 -> 256 channels (prepare for upsampling)

        # --- Decoder ---
        # Upsampling block 1 (Input: 256 from bottleneck + 256 from down3 skip -> 512 combined)
        # Output: 128 channels, size / 4
        self.up1 = Up(512, 128)
        # Self-attention block 4 (operates at size 16x16)
        self.sa4 = SelfAttention(128, 16)
        # Upsampling block 2 (Input: 128 from up1 + 128 from down2 skip -> 256 combined)
        # Output: 64 channels, size / 2
        self.up2 = Up(256, 64)
        # Self-attention block 5 (operates at size 32x32)
        self.sa5 = SelfAttention(64, 32)
        # Upsampling block 3 (Input: 64 from up2 + 64 from down1 skip -> 128 combined)
        # Output: 64 channels, size
        self.up3 = Up(128, 64)
        # Self-attention block 6 (operates at original size 64x64)
        self.sa6 = SelfAttention(64, 64)

        # --- Output Layer ---
        # Final 1x1 convolution to map features to the desired output channels (c_out)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        Generates sinusoidal positional encoding for the time step 't'.
        This is a common technique to provide the model with information about the current noise level/time step.
        Args:
            t (torch.Tensor): Tensor of time steps, shape (batch_size, 1).
            channels (int): The dimension of the encoding vector (must be even).
        Returns:
            torch.Tensor: The positional encoding tensor, shape (batch_size, channels).
        """
        # Calculate the inverse frequencies for the sine and cosine components
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels) # Use torch.arange with device
        )
        # Calculate sine and cosine components
        # t.repeat(1, channels // 2) repeats the time step for each frequency dimension
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # Concatenate the sine and cosine components
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        """
        Forward pass for the U-Net.
        Args:
            x (torch.Tensor): Input image tensor (batch, c_in, height, width).
            t (torch.Tensor): Time step tensor (batch,).
        Returns:
            torch.Tensor: Output tensor, typically predicted noise (batch, c_out, height, width).
        """
        # --- Time Embedding ---
        # Convert time steps to float and add an extra dimension -> (batch, 1)
        t = t.unsqueeze(-1).type(torch.float)
        # Generate positional encoding for the time steps
        t = self.pos_encoding(t, self.time_dim) # Shape: (batch, time_dim)

        # --- Encoder Path ---
        x1 = self.inc(x)    # Initial convolution
        x2 = self.down1(x1, t) # Downsample + time emb
        x2 = self.sa1(x2)   # Self-attention
        x3 = self.down2(x2, t) # Downsample + time emb
        x3 = self.sa2(x3)   # Self-attention
        x4 = self.down3(x3, t) # Downsample + time emb
        x4 = self.sa3(x4)   # Self-attention

        # --- Bottleneck ---
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # --- Decoder Path ---
        # Note the skip connections (x3, x2, x1) passed to the Up blocks
        x = self.up1(x4, x3, t) # Upsample + skip + time emb
        x = self.sa4(x)        # Self-attention
        x = self.up2(x, x2, t) # Upsample + skip + time emb
        x = self.sa5(x)        # Self-attention
        x = self.up3(x, x1, t) # Upsample + skip + time emb
        x = self.sa6(x)        # Self-attention

        # --- Output ---
        output = self.outc(x) # Final 1x1 convolution
        return output


# --- Conditional U-Net Architecture ---
# Extends the U-Net to incorporate conditional information, such as class labels.
# This allows generating images conditioned on specific attributes.
class UNet_conditional(nn.Module):
    """A U-Net model conditioned on class labels, in addition to time."""
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        """
        Initializes the conditional U-Net model.
        Args:
            c_in (int, optional): Number of input channels. Defaults to 3.
            c_out (int, optional): Number of output channels. Defaults to 3.
            time_dim (int, optional): Dimension of the time embedding vector. Defaults to 256.
            num_classes (int, optional): Number of classes for conditioning. If None, operates like unconditional UNet. Defaults to None.
            device (str, optional): Device ('cuda' or 'cpu'). Defaults to "cuda".
        """
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # --- Encoder Layers (Identical to UNet) ---
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # --- Bottleneck Layers (Identical to UNet) ---
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # --- Decoder Layers (Identical to UNet) ---
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        # --- Conditional Embedding ---
        # If num_classes is provided, create an embedding layer for class labels.
        # This layer maps each class index to a vector of size time_dim.
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim).to(device) # Move to the specified device

    # Positional encoding function (Identical to UNet)
    def pos_encoding(self, t, channels):
        """Generates sinusoidal positional encoding for the time step 't'."""
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        """
        Forward pass for the conditional U-Net.
        Args:
            x (torch.Tensor): Input image tensor (batch, c_in, height, width).
            t (torch.Tensor): Time step tensor (batch,).
            y (torch.Tensor): Class label tensor (batch,). Can be None for unconditional case during training (classifier-free guidance).
        Returns:
            torch.Tensor: Output tensor (batch, c_out, height, width).
        """
        # --- Time Embedding ---
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Shape: (batch, time_dim)

        # --- Conditional Embedding Addition ---
        # If class labels 'y' are provided and the embedding layer exists
        if y is not None and hasattr(self, 'label_emb'):
            # Get the embedding for each label in the batch
            label_embeddings = self.label_emb(y) # Shape: (batch, time_dim)
            # Add the label embedding to the time embedding.
            # This combines time and class information into a single conditioning vector.
            t += label_embeddings

        # --- Encoder Path (Pass combined embedding 't' to Down blocks) ---
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # --- Bottleneck ---
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # --- Decoder Path (Pass combined embedding 't' to Up blocks) ---
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        # --- Output ---
        output = self.outc(x)
        return output


# --- Main execution block ---
# This code runs only when the script is executed directly (not imported as a module).
if __name__ == '__main__':
    from torch.amp import autocast
    # --- Example Usage ---

    # check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device= torch.device('cpu') # Uncomment this line to force CPU usage

    # Print the device being used
    print("Using device:", device)


    # Instantiate the conditional U-Net model
    # net = UNet(device="cpu") # Example for unconditional UNet
    net = UNet_conditional(num_classes=10, device=device).to(device) # Create a conditional model for 10 classes, running on CPU

    # Calculate and print the total number of parameters in the model
    print("Total parameters:", sum([p.numel() for p in net.parameters()]))

    # --- Prepare Dummy Input Data ---
    # Create a batch of 3 random input images (3 channels, 64x64 pixels)
    x = torch.randn(1, 3, 64, 64) # Shape: (batch_size, channels, height, width)
    x = x.to(device) # Move the tensor to the specified device (CPU or GPU)


    # Create a batch of time steps (long integers). Here, all samples have time step 500.
    # .new_tensor() creates a tensor on the same device and with the same dtype as x (unless specified otherwise)
    t = x.new_tensor([500] * x.shape[0]).long() # Shape: (batch_size,)

    # Create a batch of class labels (long integers). Here, all samples have label 1.
    y = x.new_tensor([1] * x.shape[0]).long() # Shape: (batch_size,)

    # --- Perform Forward Pass ---
    # Pass the dummy data through the network

    with autocast(device_type=device.type):
        output = net(x, t, y)

    # --- Print Output Shape ---
    # Print the shape of the output tensor. It should match the input spatial dimensions and output channels.
    print("Output shape:", output.shape) # Expected: torch.Size([3, 3, 64, 64])
