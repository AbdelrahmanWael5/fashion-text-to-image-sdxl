# 👔 Fashion Text-to-Image Generation with SDXL

A fashion-focused text-to-image generation system built by fine-tuning Stable Diffusion XL (SDXL) on a custom apparel dataset created from VITON-HD images and automatically generated captions.

---

## Overview

This project focuses on generating high-quality fashion images from natural language prompts.

To build a specialized fashion generation model, I first generated descriptive captions for clothing images using **Qwen2.5-VL**. The generated captions were then cleaned and preprocessed before constructing a custom image-text dataset used for **SDXL fine-tuning**.

The project covers the complete workflow of a modern multimodal generative AI pipeline, including data preparation, caption generation, diffusion model fine-tuning, and deployment through an interactive demo.

---

## Dataset

This project uses the **VITON-HD** dataset, a high-resolution virtual try-on dataset widely used in fashion-related computer vision research.

Official repository:

https://github.com/shadow2496/VITON-HD

Instead of relying on manually provided descriptions, captions were automatically generated for the clothing images using **Qwen2.5-VL**. These captions were subsequently cleaned, normalized, and filtered to improve quality and consistency before being used for training.

The processed caption files are available under:

```text
├── data/
│   ├── train/
│   │   └── train.jsonl
│   ├── val/
│   │   └── val.jsonl
│   └── test/
│       └── test.jsonl
```

---

## Data Split

The original VITON-HD dataset contains:

| Split        | Samples |
| ------------ | ------: |
| Training Set |  11,647 |
| Test Set     |   2,032 |

To create a validation set for model development and monitoring, a subset of **1,165 samples** was held out from the original training split.

The final dataset configuration used throughout training was:

| Split          | Samples |
| -------------- | ------: |
| Training Set   |  10,482 |
| Validation Set |   1,165 |
| Test Set       |   2,032 |

---

## Methodology

```text
VITON-HD Images
        ↓
Qwen2.5-VL Caption Generation
        ↓
Caption Cleaning & Preprocessing
        ↓
Custom Image-Text Dataset Creation
        ↓
Train / Validation / Test Split
        ↓
SDXL Fine-Tuning
        ↓
Fashion Image Generation
```

---

## Project Structure

```text
fashion-text-to-image-sdxl/

├── app.py
├── inference.py

├── data/
│   ├── train/
│   │   └── train.jsonl
│   ├── val/
│   │   └── val.jsonl
│   └── test/
│       └── test.jsonl

└── src/
    ├── dataset/
    │   └── vitonhd.py
    │
    ├── preprocessing/
    │   ├── generate_captions.ipynb
    │   ├── prepare_captions.ipynb
    │   ├── train_captions.jsonl
    │   └── test_captions.jsonl
    │
    ├── monitoring/
    │
    └── train.ipynb
```

---

## Features

* Automatic caption generation using Qwen2.5-VL
* Caption cleaning and preprocessing pipeline
* Custom fashion image-text dataset creation
* SDXL fine-tuning for apparel generation
* Fashion-focused text-to-image synthesis
* Interactive Streamlit inference demo

---

## Tech Stack

* PyTorch
* Hugging Face Transformers
* Diffusers
* Qwen2.5-VL
* Stable Diffusion XL (SDXL)
* Streamlit

---

## Example Prompts

* Luxury black leather jacket
* Oversized hoodie
* Green denim jacket
* Formal black shirt
* Floral t-shirt

---

## Demo

The repository includes an interactive Streamlit application for real-time image generation.

A demonstration video is available in the corresponding LinkedIn post.
Link: 
https://www.linkedin.com/posts/abdelrahman-wael-ai_ai-generativeai-computervision-activity-7472330762009231362-vJJp?utm_source=share&utm_medium=member_desktop&rcm=ACoAAESxZ4gBL6KBYIJCpRjAZ_LaewDhyTR2aks

---

## Future Improvements

* Training on larger fashion datasets
* Advanced prompt engineering
* Model deployment on cloud infrastructure

---

## Acknowledgments

* Stable Diffusion XL (SDXL)
* Qwen2.5-VL
* VITON-HD Dataset
* Hugging Face Diffusers
