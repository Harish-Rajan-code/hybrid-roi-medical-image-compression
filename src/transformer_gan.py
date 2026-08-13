

import os
import math
import random
import gc
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim
)

import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42

IMAGE_SIZE = 256
BATCH_SIZE = 8

LEARNING_RATE_GENERATOR = 1e-4
LEARNING_RATE_DISCRIMINATOR = 1e-4

NUM_EPOCHS = 40
GAN_WARMUP_EPOCHS = 5

NUM_WORKERS = 2

LAMBDA_ADV = 0.002
LAMBDA_RATE = 1e-4
LAMBDA_SSIM = 0.4
LAMBDA_L1 = 0.1



BASE_CHANNELS = 32
TOKEN_DIM = 192
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed=SEED):
    """Set random seeds for reproducible training."""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# DEVICE CONFIGURATION
# ==========================================================

def get_device():
    """Return CUDA device when available, otherwise CPU."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    return device


# ==========================================================
# TRANSFORMER-GAN MODEL
# ==========================================================


class ConvEncoder(nn.Module):
 

    def __init__(
        self,
        in_ch=1,
        base_ch=BASE_CHANNELS,
        num_down=4
    ):
        super().__init__()

        layers = []
        channels = in_ch

        for i in range(num_down):

            out_ch = base_ch * (2 ** i)

            layers += [
                nn.Conv2d(
                    channels,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]

            channels = out_ch

        self.net = nn.Sequential(*layers)

        self.out_ch = channels

    def forward(self, x):
        return self.net(x)


class TransformerBottleneck(nn.Module):
 

    def __init__(
        self,
        in_ch,
        token_dim=TOKEN_DIM,
        num_layers=TRANSFORMER_LAYERS,
        num_heads=TRANSFORMER_HEADS
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_ch,
            token_dim,
            kernel_size=1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 2,
            activation="gelu",
            batch_first=True
        )

        self.trans = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.unproj = nn.Conv2d(
            token_dim,
            in_ch,
            kernel_size=1
        )

    def forward(self, x):

        batch_size, channels, height, width = x.shape

        # Project convolutional features into Transformer token space
        tokens = self.proj(x)

        # Convert feature map:
        # [B, C, H, W] -> [B, H*W, C]
        sequence = tokens.flatten(2).permute(0, 2, 1)

        # Self-attention based Transformer processing
        sequence = self.trans(sequence)

        # Convert back:
        # [B, H*W, C] -> [B, C, H, W]
        sequence = (
            sequence
            .permute(0, 2, 1)
            .contiguous()
            .view(
                batch_size,
                -1,
                height,
                width
            )
        )

        return self.unproj(sequence)


class ConvDecoder(nn.Module):

    def __init__(
        self,
        out_ch=1,
        base_ch=BASE_CHANNELS,
        num_up=4,
        in_ch=None
    ):
        super().__init__()

        if in_ch is None:
            in_ch = base_ch * (2 ** (num_up - 1))

        layers = []
        channels = in_ch

        for i in range(num_up):

            out_ch_conv = (
                base_ch * (2 ** (num_up - i - 1))
            )

            layers += [
                nn.ConvTranspose2d(
                    channels,
                    out_ch_conv,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(out_ch_conv),
                nn.ReLU(inplace=True)
            ]

            channels = out_ch_conv

        layers += [
            nn.Conv2d(
                channels,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Sigmoid()
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerCompressor(nn.Module):


    def __init__(
        self,
        base_ch=BASE_CHANNELS,
        token_dim=TOKEN_DIM
    ):
        super().__init__()

        self.enc = ConvEncoder(
            in_ch=1,
            base_ch=base_ch,
            num_down=4
        )

        self.bot = TransformerBottleneck(
            in_ch=self.enc.out_ch,
            token_dim=token_dim,
            num_layers=TRANSFORMER_LAYERS,
            num_heads=TRANSFORMER_HEADS
        )

        self.dec = ConvDecoder(
            out_ch=1,
            base_ch=base_ch,
            num_up=4,
            in_ch=self.enc.out_ch
        )

    def forward(
        self,
        x,
        training_quant=False
    ):
        batch_size, _, height, width = x.shape

        # Encoder
        latent = self.enc(x)

        # Transformer bottleneck
        latent = self.bot(latent)

        # --------------------------------------------------
        # Straight-Through Quantization
        # --------------------------------------------------

        if training_quant:

           
            quantized_latent = (
                latent
                + torch.empty_like(latent).uniform_(
                    -0.5,
                    0.5
                )
            )

        else:

           
            quantized = (
                torch.round(latent * 255.0)
                / 255.0
            )

            
            quantized_latent = (
                quantized - latent
            ).detach() + latent

        # --------------------------------------------------
        # Gaussian-Based Bit-Rate Estimation
        # --------------------------------------------------

        scale = (
            torch.std(
                latent,
                dim=[1, 2, 3],
                keepdim=True
            )
            + 1e-8
        )

        gaussian_constant = math.sqrt(
            2.0 * math.pi
        )

        probability = (
            1.0
            / (gaussian_constant * scale)
        ) * torch.exp(
            -0.5 * (latent / scale) ** 2
        )

        probability = torch.clamp(
            probability,
            min=1e-9
        )

        bits_per_element = -torch.log2(
            probability
        )

        bits_per_image = (
            bits_per_element
            .reshape(batch_size, -1)
            .sum(dim=1)
        )

        bits_per_pixel = (
            bits_per_image
            / (height * width)
        )

       
        reconstruction = self.dec(
            quantized_latent
        )

       
        return (
            reconstruction,
            quantized_latent,
            bits_per_pixel.mean(),
            {
                "scale": scale.mean()
            }
        )


class PatchDiscriminator(nn.Module):
 

    def __init__(
        self,
        in_ch=1,
        base_ch=BASE_CHANNELS,
        n_layers=3
    ):
        super().__init__()

        layers = []
        channels = in_ch

        for i in range(n_layers):

            out_ch = base_ch * (2 ** i)

            layers += [
                nn.Conv2d(
                    channels,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.LeakyReLU(
                    0.2,
                    inplace=True
                )
            ]

            channels = out_ch

        layers += [
            nn.Conv2d(
                channels,
                1,
                kernel_size=4,
                stride=1,
                padding=0
            )
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================================
# DATASET
# ==========================================================

class XRayDataset(Dataset):

    def __init__(
        self,
        folder,
        transform=None
    ):
        self.paths = []
        self.transform = transform

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Dataset directory not found: {folder}"
            )

        for subdir, _, files in os.walk(folder):

            for filename in files:

                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    self.paths.append(
                        os.path.join(
                            subdir,
                            filename
                        )
                    )

        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image = Image.open(
            self.paths[index]
        ).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image


# ==========================================================
# SSIM LOSS
# ==========================================================

def ssim_torch(img1, img2):


    mu_x = F.avg_pool2d(
        img1,
        kernel_size=3,
        stride=1,
        padding=1
    )

    mu_y = F.avg_pool2d(
        img2,
        kernel_size=3,
        stride=1,
        padding=1
    )

    sigma_x = (
        F.avg_pool2d(
            img1 * img1,
            3,
            1,
            1
        )
        - mu_x ** 2
    )

    sigma_y = (
        F.avg_pool2d(
            img2 * img2,
            3,
            1,
            1
        )
        - mu_y ** 2
    )

    sigma_xy = (
        F.avg_pool2d(
            img1 * img2,
            3,
            1,
            1
        )
        - mu_x * mu_y
    )

   
    C1 = 0.01 * 2
    C2 = 0.03 * 2

    numerator = (
        (2 * mu_x * mu_y + C1)
        * (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + C1)
        * (sigma_x + sigma_y + C2)
    )

    ssim_map = numerator / (
        denominator + 1e-8
    )

    return ssim_map.mean()


# ==========================================================
# HYBRID COMPRESSION LOSS
# ==========================================================

def hybrid_loss(
    reconstruction,
    target,
    bits_per_pixel,
    lambda_rate=LAMBDA_RATE,
    lambda_ssim=LAMBDA_SSIM,
    lambda_l1=LAMBDA_L1
):

    mse = F.mse_loss(
        reconstruction,
        target
    )

    l1 = F.l1_loss(
        reconstruction,
        target
    )

    ssim_value = ssim_torch(
        reconstruction,
        target
    )

    ssim_loss = 1.0 - ssim_value

    total = (
        mse
        + lambda_l1 * l1
        + lambda_ssim * ssim_loss
        + lambda_rate * bits_per_pixel
    )

    return (
        total,
        mse,
        l1,
        ssim_value
    )


# ==========================================================
# TRAINING
# ==========================================================

def train_epoch(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    loader,
    device,
    lambda_adv=LAMBDA_ADV,
    lambda_rate=LAMBDA_RATE,
    lambda_ssim=LAMBDA_SSIM,
    lambda_l1=LAMBDA_L1,
    use_gan=True
):
   

    generator.train()
    discriminator.train()

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    total_bits = 0.0
    total_psnr = 0.0
    total_images = 0

    for batch in loader:

        batch = batch.to(
            device,
            non_blocking=True
        )

        batch_size = batch.size(0)
        total_images += batch_size

        # --------------------------------------------------
        # Generator update
        # --------------------------------------------------

        optimizer_g.zero_grad()

        (
            reconstruction,
            _,
            bits_per_pixel,
            _
        ) = generator(
            batch,
            training_quant=True
        )

        (
            generator_loss,
            _,
            _,
            _
        ) = hybrid_loss(
            reconstruction,
            batch,
            bits_per_pixel,
            lambda_rate,
            lambda_ssim,
            lambda_l1
        )

        if use_gan:

            fake_prediction = discriminator(
                reconstruction
            )

            adversarial_loss = F.mse_loss(
                fake_prediction,
                torch.ones_like(
                    fake_prediction
                )
            )

            generator_loss = (
                generator_loss
                + lambda_adv * adversarial_loss
            )

        generator_loss.backward()
        optimizer_g.step()

        # --------------------------------------------------
        # Discriminator update
        # --------------------------------------------------

        if use_gan:

            optimizer_d.zero_grad()

            real_prediction = discriminator(
                batch
            )

            fake_prediction = discriminator(
                reconstruction.detach()
            )

            discriminator_loss = 0.5 * (
                F.mse_loss(
                    real_prediction,
                    torch.ones_like(
                        real_prediction
                    )
                )
                +
                F.mse_loss(
                    fake_prediction,
                    torch.zeros_like(
                        fake_prediction
                    )
                )
            )

            discriminator_loss.backward()
            optimizer_d.step()

        else:

            discriminator_loss = torch.tensor(
                0.0,
                device=device
            )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        total_generator_loss += (
            generator_loss.item()
            * batch_size
        )

        total_discriminator_loss += (
            discriminator_loss.item()
            * batch_size
        )

        total_bits += (
            bits_per_pixel.detach().item()
            * batch_size
        )

        reconstruction_cpu = (
            reconstruction
            .detach()
            .cpu()
            .numpy()
        )

        batch_cpu = (
            batch
            .detach()
            .cpu()
            .numpy()
        )

        for i in range(batch_size):

            total_psnr += psnr(
                batch_cpu[i, 0],
                reconstruction_cpu[i, 0],
                data_range=1.0
            )

        del (
            reconstruction,
            generator_loss,
            discriminator_loss
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if total_images == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_generator_loss / total_images,
        total_discriminator_loss / total_images,
        total_bits / total_images,
        total_psnr / total_images
    )


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(
    generator,
    loader,
    device
):

    generator.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(
                device,
                non_blocking=True
            )

            batch_size = batch.size(0)
            total_images += batch_size

            (
                reconstruction,
                _,
                _,
                _
            ) = generator(
                batch,
                training_quant=False
            )

            reconstruction_cpu = (
                reconstruction
                .cpu()
                .numpy()
            )

            batch_cpu = (
                batch
                .cpu()
                .numpy()
            )

            for i in range(batch_size):

                total_psnr += psnr(
                    batch_cpu[i, 0],
                    reconstruction_cpu[i, 0],
                    data_range=1.0
                )

                total_ssim += ssim(
                    batch_cpu[i, 0],
                    reconstruction_cpu[i, 0],
                    data_range=1.0
                )

    if total_images == 0:
        return 0.0, 0.0

    return (
        total_psnr / total_images,
        total_ssim / total_images
    )


# ==========================================================
# MODEL SAVING
# ==========================================================

def save_model(
    model,
    output_directory,
    model_name="transformer_generator"
):
  

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    model_path = os.path.join(
        output_directory,
        f"{model_name}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f"\nGenerator model saved to:\n"
        f"{model_path}"
    )

    return model_path


# ==========================================================
# VISUALIZATION
# ==========================================================

def save_reconstruction_visualization(
    generator,
    dataset,
    device,
    output_path
):
    

    if len(dataset) == 0:
        print(
            "Validation dataset is empty. "
            "Skipping visualization."
        )
        return

    generator.eval()

    image = dataset[0].unsqueeze(0).to(
        device
    )

    with torch.no_grad():

        (
            reconstruction,
            _,
            _,
            _
        ) = generator(
            image,
            training_quant=False
        )

    image_np = (
        image
        .cpu()
        .squeeze()
        .numpy()
    )

    reconstruction_np = (
        reconstruction
        .cpu()
        .squeeze()
        .numpy()
    )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    plt.figure(
        figsize=(10, 5)
    )

    plt.subplot(1, 2, 1)
    plt.imshow(
        image_np,
        cmap="gray"
    )
    plt.title("Original ROI")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(
        reconstruction_np,
        cmap="gray"
    )
    plt.title("Reconstructed ROI")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Reconstruction visualization saved to:\n"
        f"{output_path}"
    )


# ==========================================================
# MAIN TRAINING FUNCTION
# ==========================================================

def train_model(
    dataset_directory,
    output_directory,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    num_workers=NUM_WORKERS
):
   
    device = get_device()

    # ------------------------------------------------------
    # Data preprocessing
    # ------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor()
    ])

    train_path = os.path.join(
        dataset_directory,
        "train"
    )

    validation_path = os.path.join(
        dataset_directory,
        "val"
    )

    if not os.path.exists(train_path):
        raise RuntimeError(
            f"Training ROI patches not found:\n"
            f"{train_path}"
        )

    if not os.path.exists(validation_path):
        raise RuntimeError(
            f"Validation ROI patches not found:\n"
            f"{validation_path}"
        )

    train_dataset = XRayDataset(
        train_path,
        transform
    )

    validation_dataset = XRayDataset(
        validation_path,
        transform
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset contains no images."
        )

    if len(validation_dataset) == 0:
        raise RuntimeError(
            "Validation dataset contains no images."
        )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    print(
        f"Train images: {len(train_dataset)} | "
        f"Validation images: {len(validation_dataset)}"
    )

    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------

    generator = TransformerCompressor(
        base_ch=BASE_CHANNELS,
        token_dim=TOKEN_DIM
    ).to(device)

    discriminator = PatchDiscriminator(
        in_ch=1,
        base_ch=BASE_CHANNELS,
        n_layers=3
    ).to(device)

    # ------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE_GENERATOR
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE_DISCRIMINATOR
    )

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRANSFORMER-GAN ROI COMPRESSION TRAINING")
    print("=" * 60)

    for epoch in range(num_epochs):

        # GAN training begins after the warm-up period.
        use_gan = (
            epoch >= GAN_WARMUP_EPOCHS
        )

        (
            generator_loss,
            discriminator_loss,
            bpp,
            train_psnr
        ) = train_epoch(
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            train_loader,
            device,
            lambda_adv=LAMBDA_ADV,
            lambda_rate=LAMBDA_RATE,
            lambda_ssim=LAMBDA_SSIM,
            lambda_l1=LAMBDA_L1,
            use_gan=use_gan
        )

        val_psnr, val_ssim = evaluate(
            generator,
            validation_loader,
            device
        )

        mode = (
            "GAN"
            if use_gan
            else "Reconstruction"
        )

        print(
            f"[{mode}] "
            f"Epoch {epoch + 1:02d}/{num_epochs}: "
            f"G={generator_loss:.5f}, "
            f"D={discriminator_loss:.5f}, "
            f"BPP={bpp:.4f}, "
            f"TrainPSNR={train_psnr:.2f}, "
            f"ValPSNR={val_psnr:.2f}, "
            f"ValSSIM={val_ssim:.4f}"
        )

    # ------------------------------------------------------
    # Save trained model
    # ------------------------------------------------------

    model_path = save_model(
        generator,
        output_directory,
        model_name="transformer_generator"
    )

    # ------------------------------------------------------
    # Save reconstruction visualization
    # ------------------------------------------------------

    visualization_path = os.path.join(
        output_directory,
        "reconstruction_visualization.png"
    )

    save_reconstruction_visualization(
        generator,
        validation_dataset,
        device,
        visualization_path
    )

    print("\nTransformer-GAN training complete.")

    # Cleanup
    del (
        generator,
        discriminator,
        train_loader,
        validation_loader
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return model_path


# ==========================================================
# COMMAND-LINE INTERFACE
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a Transformer-GAN based "
            "ROI image compression model."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Path to the ROI patch dataset. "
            "It should contain train/ and val/ folders."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help=(
            "Directory where the trained model and "
            "visualization will be saved."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of DataLoader workers."
    )

    args = parser.parse_args()

    set_seed()

    train_model(
        dataset_directory=args.dataset,
        output_directory=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.workers
    )


# ==========================================================
# SCRIPT ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
