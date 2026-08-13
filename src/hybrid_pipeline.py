import argparse
import math
import os
import random
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 256
BOX_INPUT = 256

SEED = 42

JPEG_QUALITY = 10
MASK_THRESHOLD = 0.25
ROI_PADDING = 10


BLUR_KERNEL_SIZE = 21
BLUR_SIGMA = 5.0


BASE_CHANNELS = 32
TOKEN_DIM = 192
NUM_TRANSFORMER_LAYERS = 2
NUM_HEADS = 4


np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("HYBRID ROI / NON-ROI MEDICAL IMAGE COMPRESSION")
print("=" * 70)
print(f"Device: {DEVICE}")
print("=" * 70)


# ============================================================
# U-NET CUSTOM METRICS / LOSS
# ============================================================

def dice_coef_tf(y_true, y_pred, smooth=1e-6):
   

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (
        (2.0 * intersection + smooth)
        / (
            tf.reduce_sum(y_true_f)
            + tf.reduce_sum(y_pred_f)
            + smooth
        )
    )


def iou_metric_tf(y_true, y_pred, smooth=1e-6):
   

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    union = (
        tf.reduce_sum(y_true_f)
        + tf.reduce_sum(y_pred_f)
        - intersection
    )

    return (intersection + smooth) / (union + smooth)


def focal_tversky_loss(
    y_true,
    y_pred,
    alpha=0.7,
    beta=0.3,
    gamma=0.75,
    smooth=1e-6,
):
    

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)

    tversky = (
        (tp + smooth)
        / (
            tp
            + alpha * fn
            + beta * fp
            + smooth
        )
    )

    return tf.pow(1.0 - tversky, gamma)


# ============================================================
# ATTENTION U-NET LOADING
# ============================================================

def load_unet(model_path):
    

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"U-Net model not found:\n{model_path}"
        )

    print("\nLoading Attention U-Net...")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coef_tf": dice_coef_tf,
            "iou_metric_tf": iou_metric_tf,
            "focal_tversky_loss": focal_tversky_loss,
        },
        compile=False,
    )

    print("Attention U-Net loaded successfully.")

    return model


# ============================================================
# TRANSFORMER COMPRESSOR
# ============================================================

class ConvEncoder(nn.Module):


    def __init__(
        self,
        in_ch=1,
        base_ch=BASE_CHANNELS,
        num_down=4,
    ):
        super().__init__()

        layers = []
        channels = in_ch

        for i in range(num_down):
            out_ch = base_ch * (2 ** i)

            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        out_ch,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )

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
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_ch,
            token_dim,
            kernel_size=1,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 2,
            activation="gelu",
            batch_first=True,
        )

       
        self.trans = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.unproj = nn.Conv2d(
            token_dim,
            in_ch,
            kernel_size=1,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        tokens = self.proj(x)

        tokens = (
            tokens
            .flatten(2)
            .permute(0, 2, 1)
        )

        tokens = self.trans(tokens)

        tokens = (
            tokens
            .permute(0, 2, 1)
            .contiguous()
            .view(
                batch_size,
                -1,
                height,
                width,
            )
        )

        return self.unproj(tokens)


class ConvDecoder(nn.Module):


    def __init__(
        self,
        out_ch=1,
        base_ch=BASE_CHANNELS,
        num_up=4,
        in_ch=None,
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

            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        out_ch_conv,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_ch_conv),
                    nn.ReLU(inplace=True),
                ]
            )

            channels = out_ch_conv

        layers.extend(
            [
                nn.Conv2d(
                    channels,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Sigmoid(),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerCompressor(nn.Module):

    def __init__(
        self,
        base_ch=BASE_CHANNELS,
        token_dim=TOKEN_DIM,
    ):
        super().__init__()

        self.enc = ConvEncoder(
            in_ch=1,
            base_ch=base_ch,
            num_down=4,
        )

        self.bot = TransformerBottleneck(
            in_ch=self.enc.out_ch,
            token_dim=token_dim,
            num_layers=NUM_TRANSFORMER_LAYERS,
            num_heads=NUM_HEADS,
        )

        self.dec = ConvDecoder(
            out_ch=1,
            base_ch=base_ch,
            num_up=4,
            in_ch=self.enc.out_ch,
        )

    def forward(self, x, training_quant=False):
        batch_size, _, height, width = x.shape

      
        latent = self.enc(x)

        latent = self.bot(latent)


        if training_quant:
            quantized_latent = (
                latent
                + torch.empty_like(latent).uniform_(
                    -0.5,
                    0.5,
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

     
        scale = (
            torch.std(
                latent,
                dim=[1, 2, 3],
                keepdim=True,
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
            min=1e-9,
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
                "scale": scale.mean(),
            },
        )


# ============================================================
# TRANSFORMER MODEL LOADING
# ============================================================

def clean_state_dict(state_dict):


    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def load_transformer(model_path):

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Transformer model not found:\n{model_path}"
        )

    print("\nLoading Transformer compressor...")

    model = TransformerCompressor(
        base_ch=BASE_CHANNELS,
        token_dim=TOKEN_DIM,
    ).to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:

            state_dict = checkpoint
    else:
        raise TypeError(
            "Unsupported Transformer checkpoint format. "
            "Expected a PyTorch state_dict dictionary."
        )

    state_dict = clean_state_dict(state_dict)

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Transformer checkpoint does not match the "
            "inference architecture. No partial loading was allowed.\n\n"
            f"Original error:\n{exc}"
        ) from exc

    model.eval()

    print("Transformer compressor loaded successfully.")
    print("Checkpoint loaded with strict=True.")

    return model


# ============================================================
# IMAGE QUALITY METRICS
# ============================================================

def compute_lpips_like_metric(original, reconstructed):


    original = original.astype(np.float32) / 255.0
    reconstructed = reconstructed.astype(np.float32) / 255.0

    original_t = (
        torch.from_numpy(original)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    reconstructed_t = (
        torch.from_numpy(reconstructed)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    pool = nn.AvgPool2d(kernel_size=2)

    original_features = pool(original_t)
    reconstructed_features = pool(reconstructed_t)

    distance = torch.mean(
        (original_features - reconstructed_features) ** 2
    )

    return float(distance.item())


def compute_metrics(original, reconstructed):


    original_float = (
        original.astype(np.float32) / 255.0
    )

    reconstructed_float = (
        reconstructed.astype(np.float32) / 255.0
    )

    psnr_value = psnr(
        original_float,
        reconstructed_float,
        data_range=1.0,
    )

    ssim_value = ssim(
        original_float,
        reconstructed_float,
        data_range=1.0,
    )

    mse = np.mean(
        (
            original.astype(np.float32)
            - reconstructed.astype(np.float32)
        ) ** 2
    )

    mae = np.mean(
        np.abs(
            original.astype(np.float32)
            - reconstructed.astype(np.float32)
        )
    )

    perceptual_distance = compute_lpips_like_metric(
        original,
        reconstructed,
    )

    return {
        "psnr": float(psnr_value),
        "ssim": float(ssim_value),
        "perceptual_distance": perceptual_distance,
        "mse": float(mse),
        "mae": float(mae),
    }


# ============================================================
# COMPRESSION METRICS
# ============================================================

def calculate_compression_metrics(
    original_shape,
    compressed_size_bytes,
):

    height, width = original_shape[:2]

    original_size_bytes = height * width

    if compressed_size_bytes <= 0:
        return 0.0, 0.0

    compression_ratio = (
        original_size_bytes
        / compressed_size_bytes
    )

    bpp = (
        compressed_size_bytes * 8.0
        / (height * width)
    )

    return (
        float(compression_ratio),
        float(bpp),
    )


# ============================================================
# ROI MASK PROCESSING
# ============================================================

def refine_mask(mask):


    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return mask


def create_soft_mask(binary_mask):


    kernel = (
        BLUR_KERNEL_SIZE,
        BLUR_KERNEL_SIZE,
    )

    soft_mask = cv2.GaussianBlur(
        binary_mask.astype(np.float32),
        kernel,
        BLUR_SIGMA,
    )

    soft_mask /= 255.0

    return np.clip(
        soft_mask,
        0.0,
        1.0,
    )


# ============================================================
# ROI EXTRACTION
# ============================================================

def detect_roi(unet, image):


    height, width = image.shape

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    equalized = clahe.apply(image)

    resized = cv2.resize(
        equalized,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    model_input = (
        resized.astype(np.float32) / 255.0
    )

    model_input = model_input[
        None,
        ...,
        None,
    ]

    prediction = unet.predict(
        model_input,
        verbose=0,
    )[0, :, :, 0]

    mask = (
        prediction > MASK_THRESHOLD
    ).astype(np.uint8)

    mask = refine_mask(mask)

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    ys, xs = np.where(mask > 0)

    if len(ys) == 0:
        return None, None, 0.0

    x1 = max(
        0,
        int(xs.min()) - ROI_PADDING,
    )

    y1 = max(
        0,
        int(ys.min()) - ROI_PADDING,
    )

    x2 = min(
        width,
        int(xs.max()) + ROI_PADDING + 1,
    )

    y2 = min(
        height,
        int(ys.max()) + ROI_PADDING + 1,
    )

    roi_coverage = (
        np.sum(mask)
        / (height * width)
        * 100.0
    )

    return (
        mask,
        (x1, y1, x2, y2),
        float(roi_coverage),
    )


# ============================================================
# ROI TRANSFORMER RECONSTRUCTION
# ============================================================

def reconstruct_roi(
    transformer,
    roi,
):


    roi_resized = cv2.resize(
        roi,
        (BOX_INPUT, BOX_INPUT),
        interpolation=cv2.INTER_AREA,
    )

    roi_tensor = (
        torch.from_numpy(
            roi_resized.astype(np.float32) / 255.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        (
            reconstructed,
            quantized_latent,
            bits_per_pixel,
            metadata,
        ) = transformer(
            roi_tensor,
            training_quant=False,
        )

    reconstructed = (
        reconstructed[0, 0]
        .cpu()
        .numpy()
    )

    reconstructed = np.clip(
        reconstructed * 255.0,
        0,
        255,
    ).astype(np.uint8)

    reconstructed = cv2.resize(
        reconstructed,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    return (
        reconstructed,
        quantized_latent,
        bits_per_pixel,
        metadata,
    )


# ============================================================
# MAIN IMAGE PROCESSING
# ============================================================

def process_single_image(
    image_path,
    unet,
    transformer,
    output_dir,
):

    print("\n" + "=" * 70)
    print("PROCESSING IMAGE")
    print("=" * 70)

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    height, width = image.shape

    print(
        f"Image size: {width} × {height}"
    )

    # --------------------------------------------------------
    # STEP 1: ROI SEGMENTATION
    # --------------------------------------------------------

    print("\n[1/5] Detecting ROI...")

    mask, roi_box, roi_coverage = detect_roi(
        unet,
        image,
    )

    if mask is None:
        raise RuntimeError(
            "Attention U-Net did not detect an ROI."
        )

    x1, y1, x2, y2 = roi_box

    print(
        f"ROI coverage: {roi_coverage:.2f}%"
    )

    print(
        f"ROI bounding box: "
        f"({x1}, {y1}) → ({x2}, {y2})"
    )

    # --------------------------------------------------------
    # STEP 2: EXTRACT ROI
    # --------------------------------------------------------

    print("\n[2/5] Extracting ROI...")

    roi = image[
        y1:y2,
        x1:x2,
    ]

    if roi.size == 0:
        raise RuntimeError(
            "Extracted ROI is empty."
        )

    # --------------------------------------------------------
    # STEP 3: TRANSFORMER-GAN ROI RECONSTRUCTION
    # --------------------------------------------------------

    print(
        "\n[3/5] Reconstructing ROI "
        "with Transformer compressor..."
    )

    (
        reconstructed_roi,
        quantized_latent,
        roi_bpp_tensor,
        transformer_metadata,
    ) = reconstruct_roi(
        transformer,
        roi,
    )

    roi_bpp_estimate = float(
        roi_bpp_tensor.detach().item()
    )

    roi_metrics = compute_metrics(
        roi,
        reconstructed_roi,
    )

    print(
        f"ROI PSNR: {roi_metrics['psnr']:.2f} dB"
    )

    print(
        f"ROI SSIM: {roi_metrics['ssim']:.4f}"
    )

    print(
        f"Estimated ROI BPP: "
        f"{roi_bpp_estimate:.4f}"
    )

    # --------------------------------------------------------
    # STEP 4: JPEG NON-ROI COMPRESSION
    # --------------------------------------------------------

    print(
        "\n[4/5] Compressing non-ROI "
        f"with JPEG Q={JPEG_QUALITY}..."
    )

    # Use the binary mask to preserve only the background.
    background = image.copy()

    background[mask > 0] = 0

    # Encode background directly in memory.
    success, encoded_background = cv2.imencode(
        ".jpg",
        background,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if not success:
        raise RuntimeError(
            "JPEG background encoding failed."
        )

    compressed_background = cv2.imdecode(
        encoded_background,
        cv2.IMREAD_GRAYSCALE,
    )

    if compressed_background is None:
        raise RuntimeError(
            "JPEG background decoding failed."
        )

    background_metrics = compute_metrics(
        background,
        compressed_background,
    )

    background_size_bytes = (
        len(encoded_background)
    )

    # --------------------------------------------------------
    # STEP 5: SMOOTH HYBRID FUSION
    # --------------------------------------------------------

    print(
        "\n[5/5] Creating hybrid reconstruction..."
    )

    roi_canvas = np.zeros_like(
        image,
        dtype=np.uint8,
    )

    roi_canvas[
        y1:y2,
        x1:x2,
    ] = reconstructed_roi

    soft_mask = create_soft_mask(
        mask * 255
    )

  
    hybrid = (
        compressed_background.astype(np.float32)
        * (1.0 - soft_mask)
        + roi_canvas.astype(np.float32)
        * soft_mask
    )

    hybrid = np.clip(
        hybrid,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    overall_metrics = compute_metrics(
        image,
        hybrid,
    )

    # --------------------------------------------------------
    # COMPRESSION SIZE ESTIMATION
    # --------------------------------------------------------

 
    roi_pixel_count = roi.size

    estimated_roi_bytes = (
        roi_bpp_estimate
        * roi_pixel_count
        / 8.0
    )

    estimated_total_bytes = (
        background_size_bytes
        + estimated_roi_bytes
    )

    overall_cr, overall_bpp = (
        calculate_compression_metrics(
            image.shape,
            estimated_total_bytes,
        )
    )

    roi_cr = (
        roi.size
        / estimated_roi_bytes
        if estimated_roi_bytes > 0
        else 0.0
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_image_path = os.path.join(
        output_dir,
        f"{base_name}_hybrid.png",
    )

    mask_path = os.path.join(
        output_dir,
        f"{base_name}_mask.png",
    )

    roi_path = os.path.join(
        output_dir,
        f"{base_name}_roi_reconstructed.png",
    )

    background_path = os.path.join(
        output_dir,
        f"{base_name}_background.jpg",
    )

    cv2.imwrite(
        output_image_path,
        hybrid,
    )

    cv2.imwrite(
        mask_path,
        mask * 255,
    )

    cv2.imwrite(
        roi_path,
        reconstructed_roi,
    )

    cv2.imwrite(
        background_path,
        compressed_background,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    results = {
        "filename": os.path.basename(image_path),
        "original": image,
        "mask": mask,
        "roi": roi,
        "roi_reconstructed": reconstructed_roi,
        "background": compressed_background,
        "hybrid": hybrid,
        "roi_box": roi_box,
        "roi_coverage": roi_coverage,
        "roi_metrics": roi_metrics,
        "background_metrics": background_metrics,
        "overall_metrics": overall_metrics,
        "roi_bpp": roi_bpp_estimate,
        "roi_cr": roi_cr,
        "transformer_scale": float(
            transformer_metadata["scale"].detach().item()
        ),
        "background_size_bytes": background_size_bytes,
        "estimated_roi_bytes": estimated_roi_bytes,
        "estimated_total_bytes": estimated_total_bytes,
        "overall_cr": overall_cr,
        "overall_bpp": overall_bpp,
        "output_image": output_image_path,
        "mask_path": mask_path,
        "roi_path": roi_path,
        "background_path": background_path,
    }

    return results


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(results):
    """Print all final evaluation metrics."""

    print("\n")
    print("=" * 70)
    print("FINAL HYBRID COMPRESSION RESULTS")
    print("=" * 70)

    print(
        f"Image: {results['filename']}"
    )

    print(
        f"ROI Coverage: "
        f"{results['roi_coverage']:.2f}%"
    )

    print(
        f"ROI Bounding Box: "
        f"{results['roi_box']}"
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    roi = results["roi_metrics"]

    print("\nROI REGION")
    print("-" * 70)

    print(
        f"PSNR : {roi['psnr']:.2f} dB"
    )

    print(
        f"SSIM : {roi['ssim']:.4f}"
    )

    print(
        f"MSE  : {roi['mse']:.4f}"
    )

    print(
        f"MAE  : {roi['mae']:.4f}"
    )

    print(
        f"Perceptual Distance : "
        f"{roi['perceptual_distance']:.6f}"
    )

    print(
        f"Transformer BPP : "
        f"{results['roi_bpp']:.4f}"
    )

    print(
        f"Estimated CR  : "
        f"{results['roi_cr']:.2f}×"
    )

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    background = results[
        "background_metrics"
    ]

    print("\nNON-ROI / BACKGROUND")
    print("-" * 70)

    print(
        f"PSNR : {background['psnr']:.2f} dB"
    )

    print(
        f"SSIM : {background['ssim']:.4f}"
    )

    print(
        f"MSE  : {background['mse']:.4f}"
    )

    print(
        f"MAE  : {background['mae']:.4f}"
    )

    print(
        f"Perceptual Distance : "
        f"{background['perceptual_distance']:.6f}"
    )

    print(
        f"JPEG Size : "
        f"{results['background_size_bytes']} bytes"
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall = results[
        "overall_metrics"
    ]

    print("\nOVERALL HYBRID IMAGE")
    print("-" * 70)

    print(
        f"PSNR : {overall['psnr']:.2f} dB"
    )

    print(
        f"SSIM : {overall['ssim']:.4f}"
    )

    print(
        f"MSE  : {overall['mse']:.4f}"
    )

    print(
        f"MAE  : {overall['mae']:.4f}"
    )

    print(
        f"Perceptual Distance : "
        f"{overall['perceptual_distance']:.6f}"
    )

    print(
        f"Estimated Compression Ratio : "
        f"{results['overall_cr']:.2f}×"
    )

    print(
        f"Estimated BPP : "
        f"{results['overall_bpp']:.4f}"
    )

    print(
        f"\nOutput saved to:"
        f"\n{results['output_image']}"
    )

    print("=" * 70)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_pipeline(results):
   
    fig = plt.figure(
        figsize=(20, 5)
    )

    fig.suptitle(
        "Hybrid ROI / Non-ROI Medical Image Compression",
        fontsize=15,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    ax1 = plt.subplot(1, 5, 1)

    ax1.imshow(
        results["original"],
        cmap="gray",
    )

    ax1.set_title("Original")
    ax1.axis("off")

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    ax2 = plt.subplot(1, 5, 2)

    ax2.imshow(
        results["mask"],
        cmap="gray",
    )

    ax2.set_title("Predicted ROI Mask")
    ax2.axis("off")

    # --------------------------------------------------------
    # ROI Reconstruction
    # --------------------------------------------------------

    ax3 = plt.subplot(1, 5, 3)

    ax3.imshow(
        results["roi_reconstructed"],
        cmap="gray",
    )

    ax3.set_title("Transformer ROI")
    ax3.axis("off")

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    ax4 = plt.subplot(1, 5, 4)

    ax4.imshow(
        results["background"],
        cmap="gray",
    )

    ax4.set_title(
        f"JPEG Background Q={JPEG_QUALITY}"
    )

    ax4.axis("off")

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    ax5 = plt.subplot(1, 5, 5)

    ax5.imshow(
        results["hybrid"],
        cmap="gray",
    )

    ax5.set_title(
        "Final Hybrid"
    )

    ax5.axis("off")

    plt.tight_layout()

    plt.show()


def visualize_detailed(results):
    

    fig = plt.figure(
        figsize=(16, 5)
    )

    fig.suptitle(
        "ROI vs Non-ROI Compression",
        fontsize=15,
        fontweight="bold",
    )

    ax1 = plt.subplot(1, 4, 1)

    ax1.imshow(
        results["roi"],
        cmap="gray",
    )

    ax1.set_title("Original ROI")
    ax1.axis("off")

    ax2 = plt.subplot(1, 4, 2)

    ax2.imshow(
        results["roi_reconstructed"],
        cmap="gray",
    )

    ax2.set_title(
        "Transformer Reconstruction"
    )

    ax2.axis("off")

    ax3 = plt.subplot(1, 4, 3)

    ax3.imshow(
        results["background"],
        cmap="gray",
    )

    ax3.set_title(
        f"JPEG Background Q={JPEG_QUALITY}"
    )

    ax3.axis("off")

    ax4 = plt.subplot(1, 4, 4)

    ax4.imshow(
        results["hybrid"],
        cmap="gray",
    )

    ax4.set_title(
        f"Hybrid\n"
        f"SSIM={results['overall_metrics']['ssim']:.4f}"
    )

    ax4.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid ROI / Non-ROI medical image "
            "compression using Attention U-Net "
            "and Transformer-GAN."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input X-ray image.",
    )

    parser.add_argument(
        "--unet",
        required=True,
        help="Path to trained Attention U-Net (.keras).",
    )

    parser.add_argument(
        "--transformer",
        required=True,
        help="Path to trained Transformer checkpoint (.pth).",
    )

    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory for generated outputs.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=JPEG_QUALITY,
        help="JPEG quality for non-ROI region.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    global JPEG_QUALITY

    args = parse_arguments()

    JPEG_QUALITY = max(
        1,
        min(100, args.jpeg_quality),
    )

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not os.path.isfile(args.image):
        raise FileNotFoundError(
            f"Input image not found:\n{args.image}"
        )

    if not os.path.isfile(args.unet):
        raise FileNotFoundError(
            f"U-Net model not found:\n{args.unet}"
        )

    if not os.path.isfile(args.transformer):
        raise FileNotFoundError(
            f"Transformer model not found:\n"
            f"{args.transformer}"
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    unet = load_unet(
        args.unet
    )

    transformer = load_transformer(
        args.transformer
    )

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    results = process_single_image(
        image_path=args.image,
        unet=unet,
        transformer=transformer,
        output_dir=args.output,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_metrics(results)

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    visualize_pipeline(results)

    visualize_detailed(results)

    print(
        "\nHybrid compression pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
