"""
ROI Segmentation using Attention U-Net

This module implements the ROI segmentation stage of the
Hybrid ROI and Non-ROI Medical Image Compression Framework.

The model uses:
- Residual convolution blocks
- Attention gates
- CLAHE preprocessing
- Focal Tversky loss
- Dice and IoU evaluation

The trained model can subsequently be used by the hybrid
compression pipeline to identify diagnostically important regions.
"""

import os
import random
import gc
import argparse
from glob import glob

import cv2
import numpy as np
import scipy.ndimage as ndi
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from keras import ops


# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42
IMG_SIZE = 256
EPOCHS_UNET = 40
BATCH_SIZE = 4
LEARNING_RATE_UNET = 2e-4

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ==========================================================
# GPU CONFIGURATION
# ==========================================================

def configure_gpu():
    """Enable GPU memory growth when a compatible GPU is available."""
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print(f"GPU available: {len(gpus)} device(s) detected.")

        except RuntimeError as error:
            print(f"GPU configuration warning: {error}")
    else:
        print("No GPU detected. Training will use the available CPU.")


# ==========================================================
# MASK REFINEMENT
# ==========================================================

def refine_mask(mask, closing_iter=3, min_size=500):
    """
    Refine a binary ROI mask using morphological operations.

    Operations:
    - Morphological closing
    - Hole filling
    - Removal of small connected components
    - Morphological opening
    """

    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=closing_iter
    )

    # Fill internal holes
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    # Remove small connected components
    labeled, num_components = ndi.label(mask)

    for component_id in range(1, num_components + 1):
        component = labeled == component_id

        if component.sum() < min_size:
            mask[component] = 0

    # Remove small irregular structures
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return mask


# ==========================================================
# ATTENTION GATE
# ==========================================================

def attention_gate(x, g, inter_channels):
    """
    Attention gate used in the decoder of the Attention U-Net.

    Parameters
    ----------
    x : Tensor
        Encoder feature map.
    g : Tensor
        Decoder/gating feature map.
    inter_channels : int
        Number of intermediate channels.

    Returns
    -------
    Tensor
        Attention-weighted encoder feature map.
    """

    theta_x = layers.Conv2D(
        inter_channels,
        (1, 1),
        padding="same"
    )(x)

    phi_g = layers.Conv2D(
        inter_channels,
        (1, 1),
        padding="same"
    )(g)

    add_xg = layers.Add()([theta_x, phi_g])
    act_xg = layers.Activation("relu")(add_xg)

    psi = layers.Conv2D(
        1,
        (1, 1),
        padding="same"
    )(act_xg)

    psi = layers.Activation("sigmoid")(psi)

    psi = layers.Resizing(
        height=x.shape[1],
        width=x.shape[2],
        interpolation="bilinear"
    )(psi)

    out = layers.Multiply()([x, psi])

    return out


# ==========================================================
# RESIDUAL CONVOLUTIONAL BLOCK
# ==========================================================

def residual_conv_block(x, filters, use_batchnorm=True):
    """
    Residual convolutional block used throughout the U-Net.
    """

    shortcut = x

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same"
    )(x)

    if use_batchnorm:
        x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same"
    )(x)

    if use_batchnorm:
        x = layers.BatchNormalization()(x)

    # Match the number of channels in the residual connection.
    # Static shape is used here instead of ops.shape() so that
    # the comparison works correctly during model construction.
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            (1, 1),
            padding="same"
        )(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    return x


# ==========================================================
# ATTENTION U-NET MODEL
# ==========================================================

def attention_unet(input_size=(256, 256, 1)):
    """
    Build an Attention U-Net with residual connections.
    """

    inputs = layers.Input(input_size)

    # ---------------- Encoder ----------------

    c1 = residual_conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)

    c2 = residual_conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)

    c3 = residual_conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)

    c4 = residual_conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.2)(p4)

    # ---------------- Bottleneck ----------------

    c5 = residual_conv_block(p4, 1024)
    c5 = layers.Dropout(0.3)(c5)

    # ---------------- Decoder ----------------

    u6 = layers.Conv2DTranspose(
        512,
        (2, 2),
        strides=2,
        padding="same"
    )(c5)

    att4 = attention_gate(c4, u6, 256)
    u6 = layers.Concatenate()([u6, att4])
    c6 = residual_conv_block(u6, 512)

    u7 = layers.Conv2DTranspose(
        256,
        (2, 2),
        strides=2,
        padding="same"
    )(c6)

    att3 = attention_gate(c3, u7, 128)
    u7 = layers.Concatenate()([u7, att3])
    c7 = residual_conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(
        128,
        (2, 2),
        strides=2,
        padding="same"
    )(c7)

    att2 = attention_gate(c2, u8, 64)
    u8 = layers.Concatenate()([u8, att2])
    c8 = residual_conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(
        64,
        (2, 2),
        strides=2,
        padding="same"
    )(c8)

    att1 = attention_gate(c1, u9, 32)
    u9 = layers.Concatenate()([u9, att1])
    c9 = residual_conv_block(u9, 64)

    outputs = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid"
    )(c9)

    return models.Model(inputs, outputs)


# ==========================================================
# METRICS
# ==========================================================

def dice_coef_tf(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient."""

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(
        y_true_f * y_pred_f
    )

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true_f)
        + tf.reduce_sum(y_pred_f)
        + smooth
    )


def iou_metric_tf(y_true, y_pred, smooth=1e-6):
    """Calculate Intersection over Union."""

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(
        y_true_f * y_pred_f
    )

    union = (
        tf.reduce_sum(y_true_f)
        + tf.reduce_sum(y_pred_f)
        - intersection
    )

    return (
        intersection + smooth
    ) / (
        union + smooth
    )


# ==========================================================
# FOCAL TVERSKY LOSS
# ==========================================================

def focal_tversky_loss(
    y_true,
    y_pred,
    alpha=0.7,
    beta=0.3,
    gamma=0.75,
    smooth=1e-6
):
    """Focal Tversky loss used for ROI segmentation."""

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    true_positive = tf.reduce_sum(
        y_true_f * y_pred_f
    )

    false_negative = tf.reduce_sum(
        y_true_f * (1.0 - y_pred_f)
    )

    false_positive = tf.reduce_sum(
        (1.0 - y_true_f) * y_pred_f
    )

    tversky = (
        true_positive + smooth
    ) / (
        true_positive
        + alpha * false_negative
        + beta * false_positive
        + smooth
    )

    return tf.pow(1.0 - tversky, gamma)


# ==========================================================
# DATA LOADER
# ==========================================================

def load_pairs(root, img_size=IMG_SIZE, augment=False):
    """
    Load image-mask pairs recursively from a directory.

    Image files are matched with mask files using the image
    filename stem.
    """

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset directory not found: {root}"
        )

    # Recursively collect image files.
    all_files = glob(
        os.path.join(root, "**", "*"),
        recursive=True
    )

    img_files = [
        file_path
        for file_path in all_files
        if (
            "mask" not in os.path.basename(file_path).lower()
            and file_path.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        )
    ]

    images = []
    masks = []

    for img_path in img_files:

        base = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        mask_candidates = [
            file_path
            for file_path in all_files
            if (
                base in os.path.basename(file_path)
                and "mask" in os.path.basename(file_path).lower()
                and file_path.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            )
        ]

        if not mask_candidates:
            continue

        mask_path = mask_candidates[0]

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None or mask is None:
            continue

        # CLAHE preprocessing
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        img = clahe.apply(img)

        # Resize and normalize image
        img = cv2.resize(
            img,
            (img_size, img_size)
        ).astype(np.float32) / 255.0

        # Resize and binarize mask
        mask = cv2.resize(
            mask,
            (img_size, img_size)
        )

        mask = (
            mask > 127
        ).astype(np.float32)

        images.append(
            np.expand_dims(img, -1)
        )

        masks.append(
            np.expand_dims(mask, -1)
        )

        # Horizontal flip augmentation
        if augment and random.random() > 0.5:

            img_flip = np.fliplr(img)
            mask_flip = np.fliplr(mask)

            images.append(
                np.expand_dims(img_flip, -1)
            )

            masks.append(
                np.expand_dims(mask_flip, -1)
            )

    print(
        f"Loaded {len(images)} image-mask pairs from {root}"
    )

    return (
        np.array(images),
        np.array(masks)
    )


# ==========================================================
# TRAINING
# ==========================================================

def train_unet(dataset_path, output_path):
    """
    Train the Attention U-Net and save the trained model.
    """

    print("=" * 60)
    print("PHASE 1: TRAINING ATTENTION U-NET ON ROI DATASET")
    print("=" * 60)

    # Load dataset
    X, Y = load_pairs(
        dataset_path,
        augment=True
    )

    if len(X) == 0:
        raise RuntimeError(
            "No valid image-mask pairs were found. "
            "Check the dataset directory and filename structure."
        )

    # Train-validation split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=SEED
    )

    print(
        f"Train: {len(X_train)} | "
        f"Validation: {len(X_val)}"
    )

    # Build model
    unet_model = attention_unet(
        (IMG_SIZE, IMG_SIZE, 1)
    )

    # Compile model
    unet_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE_UNET
        ),
        loss=focal_tversky_loss,
        metrics=[
            "accuracy",
            dice_coef_tf,
            iou_metric_tf
        ]
    )

    # Learning-rate scheduler
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(
        f"\nTraining U-Net for "
        f"{EPOCHS_UNET} epochs..."
    )

    history = unet_model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS_UNET,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    val_loss, val_acc, val_dice, val_iou = (
        unet_model.evaluate(
            X_val,
            Y_val,
            verbose=0
        )
    )

    print(
        "\nU-Net Final Results → "
        f"Accuracy: {val_acc:.4f}, "
        f"Dice: {val_dice:.4f}, "
        f"IoU: {val_iou:.4f}"
    )

    # Create output directory
    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # Save model
    unet_model.save(output_path)

    print(
        f"\nU-Net model saved successfully at:\n"
        f"{output_path}"
    )

    # Clean up
    tf.keras.backend.clear_session()

    del (
        unet_model,
        X,
        Y,
        X_train,
        Y_train,
        X_val,
        Y_val,
        history
    )

    gc.collect()

    print("\nU-Net phase complete.")


# ==========================================================
# MAIN
# ==========================================================

def main():
    """
    Command-line entry point for ROI segmentation training.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train an Attention U-Net for "
            "medical image ROI segmentation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Path to the directory containing "
            "image-mask pairs."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/unet_model.keras",
        help=(
            "Path where the trained U-Net model "
            "will be saved."
        )
    )

    args = parser.parse_args()

    configure_gpu()

    train_unet(
        dataset_path=args.dataset,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
