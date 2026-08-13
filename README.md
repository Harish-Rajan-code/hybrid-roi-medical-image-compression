# Hybrid ROI and Non-ROI Medical Image Compression

A hybrid deep learning framework for region-aware medical image compression that prioritizes diagnostically important regions while achieving high overall compression.

## Overview

Medical images contain regions that are more important for diagnosis than others. Applying the same level of compression to the entire image can unnecessarily degrade diagnostically important information.

This project presents a hybrid ROI-based medical image compression framework that applies different compression strategies to different regions of a chest X-ray. An Attention U-Net first identifies the diagnostically important Region of Interest (ROI). The ROI is then compressed using a Transformer-based compression network enhanced with adversarial training, while the non-ROI region is compressed using JPEG. The compressed regions are subsequently combined to reconstruct the final image.

The objective is to reduce storage and transmission requirements while preserving the structural quality of diagnostically important regions.

## Key Features

- Attention U-Net based ROI segmentation
- Residual convolution blocks and attention gates
- Transformer-based ROI compression
- PatchGAN-based adversarial training
- Rate-distortion-aware compression objective
- JPEG compression for non-ROI regions
- Hybrid ROI/non-ROI reconstruction
- Evaluation using PSNR, SSIM, LPIPS, MSE, MAE, BPP and compression ratio

## System Architecture


                    Input Chest X-Ray
                           │
                           ▼
                    Attention U-Net
                           │
                           ▼
                      ROI Mask
                     /         \
                    /           \
                   ▼             ▼
                 ROI           Non-ROI
                  │               │
                  ▼               ▼
          Transformer-GAN        JPEG
             Compression       Compression
                  │               │
                  └───────┬───────┘
                          ▼
                  Hybrid Reconstruction
                          │
                          ▼
                Compressed X-Ray Image

## **Methodology**
**1. ROI Segmentation**

An Attention U-Net is used to identify the diagnostically important lung region in the input chest X-ray.

The implementation uses residual convolution blocks and attention gates within the encoder-decoder architecture. CLAHE is applied during preprocessing to enhance local contrast. The segmentation model is trained using Focal Tversky Loss and evaluated using Dice coefficient and Intersection over Union (IoU).

**2. ROI Compression**

The extracted ROI is compressed using a Transformer-based image compression network trained with adversarial learning.

The compressor consists of a convolutional encoder, a Transformer bottleneck, and a convolutional decoder. The Transformer bottleneck uses multi-head self-attention to model spatial dependencies in the encoded representation.

A PatchGAN discriminator is used during adversarial training to encourage high-quality reconstructions. The training objective combines reconstruction error, L1 loss, SSIM-based loss, and a rate term to balance image quality and compression efficiency.

**3. Non-ROI Compression**

Regions outside the detected ROI are compressed using JPEG with a quality factor of 10. This allows more aggressive compression of regions considered less diagnostically important.

**4. Hybrid Reconstruction**

The reconstructed ROI from the Transformer-based compressor is placed back into the JPEG-compressed background to produce the final hybrid image.

The final output is evaluated against the original image using PSNR, SSIM, LPIPS, MSE, MAE, bits per pixel (BPP), and compression ratio.



**This project was developed as a three-member academic project.**

## **My Contribution**
My primary contribution was the development and training of the Transformer-GAN based ROI compression module. My work included:

Designing the convolutional encoder-decoder architecture
Implementing the Transformer bottleneck with multi-head self-attention
Implementing the PatchGAN discriminator
Developing the compression and rate-aware training objective
Implementing quantization and bit-rate estimation
Training and evaluating the compression model
Evaluating reconstructed ROI quality using PSNR and SSIM

## **Results**

The Transformer-GAN based ROI compression achieved approximately 3.5× compression for the ROI, while the complete hybrid framework achieved approximately 10× overall image compression by applying more aggressive compression to the non-ROI region.

**The framework evaluates compression quality using:**

PSNR
SSIM
LPIPS
MSE
MAE
Bits Per Pixel (BPP)
Compression Ratio (CR)

## **Dataset**

The medical image datasets used during development are not included in this repository.

The required datasets should be obtained from their respective sources and the dataset paths should be configured in the scripts before running the project.

## **Limitations**
The current implementation was developed and tested in a GPU-based research environment.
The compression pipeline relies on trained model checkpoints.
The framework is a research prototype and has not been evaluated as a clinical diagnostic system.
Broader evaluation across larger and more diverse medical imaging datasets would be required for further validation.

## **Future Work**
Development of an explicit entropy coding stage for practical compressed bitstream generation
Evaluation across larger and more diverse medical imaging datasets
Improved rate-distortion optimization
Computational optimization for resource-constrained deployment
Extension to additional medical imaging modalities
