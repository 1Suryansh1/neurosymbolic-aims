# Neuro-Symbolic Multimodal Biomass Prediction

This repository contains the codebase and final report for an advanced Neuro-Symbolic architecture designed for agricultural biomass prediction.

## Overview
Traditional deep learning models operate as "black boxes", often violating fundamental physical laws—such as predicting negative biomass or failing to sum individual mass components to the total mass.

This repository implements a **Multimodal Neuro-Symbolic Regression Model** combining:
1. **Multimodal Fusion**: A ResNet-18 image encoder and tabular metadata encoder (MLP or NODE) to fuse ground-level imagery with continuous metadata (NDVI, Pasture Height, State, Species).
2. **Logical Tensor Networks (LTN)**: A symbolic logic engine built on `LTNtorch` to enforce strict domain constraints (Additivity, Proximity, and Non-Negativity) during gradient descent without succumbing to the scale-explosion seen in naive MSE penalty methods.

### Key Finding: The Small Data Corollary
We demonstrate a critical corollary to Sutton's *Bitter Lesson*: While data-hungry, state-of-the-art architectures (like NODE with `entmax15` routing) dominate in data-rich environments, they drastically overfit in the data-starved domain of precision agriculture. In these scenarios, injecting structural inductive priors (LTN fuzzy logic) into simpler representations (MLP) fundamentally outperforms pure data-driven overparameterization.

## Project Structure
- `src/`
  - `models/`: Contains the architecture implementations (`image_encoder.py`, `metadata_encoder.py`, `node_encoder.py`, `multimodal_model.py`).
  - `losses/`: Contains standard prediction losses and the Neuro-Symbolic constraint engines (`symbolic_losses.py`, `ltn_losses.py`).
  - `train.py`: Main training loop supporting 5-fold cross-validation across multiple configurations.
  - `evaluate.py`: Standalone evaluation script for comprehensive metrics generation.
  - `run_post_analysis.py`: Script to generate t-SNE latent projections and residual error maps.
- `final_report_stuff.docx`: The finalized, comprehensive research report detailing the methodologies and findings.

## Installation
Ensure you have Python 3.10+ installed. Install the dependencies using:

```bash
pip install -r requirements.txt
