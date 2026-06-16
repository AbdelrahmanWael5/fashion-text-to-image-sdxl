# 👔 Fashion Text-to-Image Generation with SDXL

A fashion-focused text-to-image generation system built by fine-tuning Stable Diffusion XL (SDXL) on a custom apparel dataset.

## Overview

This project aims to generate high-quality fashion images from text prompts.

To build a specialized fashion generation model, I first generated descriptive captions for clothing images in the VITON-HD dataset using Qwen2.5-VL, then cleaned and preprocessed the captions before fine-tuning Stable Diffusion XL (SDXL) on the resulting image-text dataset.

---

## Pipeline

Fashion Images
→ Qwen2.5-VL Caption Generation
→ Caption Cleaning & Preprocessing
→ Image-Text Dataset Creation
→ SDXL Fine-Tuning
→ Fashion Image Generation

---

## Project Structure

```text
fashion-text-to-image-sdxl/

├── app.py
├── inference.py
├── data/
    ├── train/train.jsonl
    ├── val/val.jsonl
    ├── test/test.jsonl

└── src/
    ├── dataset/
    ├── preprocessing/
    ├── monitoring/
    └── train.ipynb
```

## Features

- Automatic caption generation using Qwen2.5-VL
- Caption preprocessing and cleaning
- Fashion-focused SDXL fine-tuning
- Text-to-image generation
- Interactive Streamlit demo

## Tech Stack

- PyTorch
- Hugging Face Transformers
- Diffusers
- Qwen2.5-VL
- Stable Diffusion XL (SDXL)
- Streamlit

## Example Prompts

- Luxury black leather jacket
- Oversized hoodie
- Green denim jacket
- Formal black shirt
- Floral t-shirt

## Demo

Demo video available in the LinkedIn post.

## Future Improvements

- LoRA fine-tuning
- Larger fashion datasets
- Better prompt engineering
- Evaluation metrics for generated images
