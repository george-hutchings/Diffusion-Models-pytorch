# Import necessary libraries from PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Exponential Moving Average (EMA) ---
# (EMA class remains unchanged)
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
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
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
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Performs one EMA update step. Can optionally delay the start of EMA.
        Args:
            ema_model (nn.Module): The moving average model.
            model (nn.Module): The current training model.
            step_start_ema (int): The training step number after which EMA updates should begin.
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Copies the parameters from the current model to the EMA model.
        Args:
            ema_model (nn.Module): The moving average model (destination).
            model (nn.Module): The current training model (source).
        """
        ema_model.load_state_dict(model.state_dict())


# --- Self Attention Block ---
# (SelfAttention class remains unchanged)
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
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Forward pass for the SelfAttention block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


# --- Double Convolution Block ---
# (DoubleConv class remains unchanged)
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
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
        Forward pass for the DoubleConv block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


# --- Downsampling Block (Encoder part of U-Net) ---
# (Down class remains unchanged)
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
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
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
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# --- Upsampling Block (Decoder part of U-Net) ---
# (Up class remains unchanged)
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
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
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
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# --- U-Net Architecture ---
# Modified for 32x32 input
class UNet(nn.Module):
    """A U-Net model architecture adapted for 32x32 images."""
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        """
        Initializes the U-Net model for 32x32 images.
        Args:
            c_in (int, optional): Number of input channels. Defaults to 3.
            c_out (int, optional): Number of output channels. Defaults to 3.
            time_dim (int, optional): Dimension of the time embedding vector. Defaults to 256.
            device (str, optional): Device to run the model on. Defaults to "cuda".
        """
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # --- Encoder ---
        self.inc = DoubleConv(c_in, 64)         # Output: 64 channels, 32x32
        self.down1 = Down(64, 128)              # Output: 128 channels, 16x16
        self.sa1 = SelfAttention(128, 16)       # <<< CHANGED SIZE: 32 -> 16
        self.down2 = Down(128, 256)             # Output: 256 channels, 8x8
        self.sa2 = SelfAttention(256, 8)        # <<< CHANGED SIZE: 16 -> 8
        self.down3 = Down(256, 256)             # Output: 256 channels, 4x4
        self.sa3 = SelfAttention(256, 4)        # <<< CHANGED SIZE: 8 -> 4

        # --- Bottleneck --- (Operates at 4x4)
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # --- Decoder ---
        self.up1 = Up(512, 128)                 # Input: 256(bot)+256(skip), Output: 128 channels, 8x8
        self.sa4 = SelfAttention(128, 8)        # <<< CHANGED SIZE: 16 -> 8
        self.up2 = Up(256, 64)                  # Input: 128(up1)+128(skip), Output: 64 channels, 16x16
        self.sa5 = SelfAttention(64, 16)        # <<< CHANGED SIZE: 32 -> 16
        self.up3 = Up(128, 64)                  # Input: 64(up2)+64(skip), Output: 64 channels, 32x32
        self.sa6 = SelfAttention(64, 32)        # <<< CHANGED SIZE: 64 -> 32

        # --- Output Layer ---
        self.outc = nn.Conv2d(64, c_out, kernel_size=1) # Output: c_out channels, 32x32

    def pos_encoding(self, t, channels):
        """Generates sinusoidal positional encoding for the time step 't'."""
        # (Positional encoding remains unchanged)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        """
        Forward pass for the U-Net.
        Args:
            x (torch.Tensor): Input image tensor (batch, c_in, 32, 32).
            t (torch.Tensor): Time step tensor (batch,).
        Returns:
            torch.Tensor: Output tensor (batch, c_out, 32, 32).
        """
        # (Forward pass logic remains the same, layers handle sizes)
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        output = self.outc(x)
        return output


# --- Conditional U-Net Architecture ---
# Modified for 32x32 input
class UNet_conditional(nn.Module):
    """A U-Net model conditioned on class labels, adapted for 32x32 images."""
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        """
        Initializes the conditional U-Net model for 32x32 images.
        Args:
            c_in (int, optional): Number of input channels. Defaults to 3.
            c_out (int, optional): Number of output channels. Defaults to 3.
            time_dim (int, optional): Dimension of the time embedding vector. Defaults to 256.
            num_classes (int, optional): Number of classes for conditioning. Defaults to None.
            device (str, optional): Device ('cuda' or 'cpu'). Defaults to "cuda".
        """
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # --- Encoder Layers --- (Identical structure to UNet, sizes adjusted)
        self.inc = DoubleConv(c_in, 64)         # Output: 64 channels, 32x32
        self.down1 = Down(64, 128)              # Output: 128 channels, 16x16
        self.sa1 = SelfAttention(128, 16)       # <<< CHANGED SIZE: 32 -> 16
        self.down2 = Down(128, 256)             # Output: 256 channels, 8x8
        self.sa2 = SelfAttention(256, 8)        # <<< CHANGED SIZE: 16 -> 8
        self.down3 = Down(256, 256)             # Output: 256 channels, 4x4
        self.sa3 = SelfAttention(256, 4)        # <<< CHANGED SIZE: 8 -> 4

        # --- Bottleneck Layers --- (Operates at 4x4)
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # --- Decoder Layers --- (Identical structure to UNet, sizes adjusted)
        self.up1 = Up(512, 128)                 # Input: 256(bot)+256(skip), Output: 128 channels, 8x8
        self.sa4 = SelfAttention(128, 8)        # <<< CHANGED SIZE: 16 -> 8
        self.up2 = Up(256, 64)                  # Input: 128(up1)+128(skip), Output: 64 channels, 16x16
        self.sa5 = SelfAttention(64, 16)        # <<< CHANGED SIZE: 32 -> 16
        self.up3 = Up(128, 64)                  # Input: 64(up2)+64(skip), Output: 64 channels, 32x32
        self.sa6 = SelfAttention(64, 32)        # <<< CHANGED SIZE: 64 -> 32
        self.outc = nn.Conv2d(64, c_out, kernel_size=1) # Output: c_out channels, 32x32

        # --- Conditional Embedding ---
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim).to(device)

    def pos_encoding(self, t, channels):
        """Generates sinusoidal positional encoding for the time step 't'."""
        # (Positional encoding remains unchanged)
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
            x (torch.Tensor): Input image tensor (batch, c_in, 32, 32).
            t (torch.Tensor): Time step tensor (batch,).
            y (torch.Tensor): Class label tensor (batch,). Can be None.
        Returns:
            torch.Tensor: Output tensor (batch, c_out, 32, 32).
        """
        # (Forward pass logic remains the same, layers handle sizes)
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None and hasattr(self, 'label_emb'):
            label_embeddings = self.label_emb(y)
            t += label_embeddings

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        output = self.outc(x)
        return output


# --- Main execution block ---
if __name__ == '__main__':
    # Import autocast only if needed here
    try:
        from torch.amp import autocast
        use_amp = True
    except ImportError:
        print("Warning: torch.amp not available. Running without autocast.")
        use_amp = False
        # Define a dummy context manager if autocast is not available
        class autocast:
            def __init__(self, device_type):
                self.device_type = device_type
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

    # --- Example Usage ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu') # Uncomment this line to force CPU usage

    print("Using device:", device)

    # Instantiate the conditional U-Net model
    # net = UNet(device=device).to(device) # Example for unconditional UNet
    net = UNet_conditional(num_classes=10, device=device).to(device) # Conditional model for 10 classes

    print("Total parameters:", sum(p.numel() for p in net.parameters()))

    # --- Prepare Dummy Input Data ---
    # Create a batch of random input images (3 channels, 32x32 pixels)
    x = torch.randn(1, 3, 32, 32)           # <<< CHANGED SIZE: 64 -> 32
    x = x.to(device)

    # Create dummy time steps and labels
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()

    # --- Perform Forward Pass ---
    # Use autocast if available (especially useful on GPU with appropriate hardware)
    # Note: autocast might have limited benefit or require adjustments on CPU
    if use_amp and device.type != 'cpu': # Typically use autocast for CUDA
         with autocast(device_type=device.type):
            output = net(x, t, y)
    else:
         # Run without autocast on CPU or if unavailable
         output = net(x, t, y)


    # --- Print Output Shape ---
    # Should match the input spatial dimensions (32x32)
    print("Output shape:", output.shape) # Expected: torch.Size([1, 3, 32, 32])