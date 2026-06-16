# PolymerLife-QSPR

PolymerLife-QSPR is a research-oriented project focused on **Quantitative Structure-Property Relationship (QSPR)** modeling for polymer materials. Using machine learning, the project predicts key physical properties such as **Thermal Conductivity (Tc)**, **Glass Transition Temperature (Tg)**, and **Density** directly from their chemical structures represented as SMILES strings.

## 🚀 Overview

The project provides tools for:
- **Property Prediction**: High-fidelity ML models for predicting polymer characteristics.
- **Explainable AI**: Utilizing SHAP to identify critical molecular fragments and descriptors that drive material properties.
- **Durability Simulations**: Analyzing how thermal properties influence the long-term retention of polymer performance under environmental stress.

## 📁 Directory Structure

- `data/`: Raw and processed polymer datasets (e.g., `extended_polymer_dataset.csv`).
- `models/`: Serialized pre-trained machine learning models (`.pkl`).
- `results/`:
    - `figures/`: Analysis plots including parity and SHAP summary charts.
    - `mfpgen/`: Visualizations of influential molecular fragments.
    - `top_n/`: Data on the most significant features for property prediction.
    - `survival_curves.py`: Script for simulating property retention over time.
- `visualization/`: Chemical structure visualization logs and utilities.
- `prediction/`, `smiles/`, `utils/`: Core modules for molecular feature engineering and inference.
- `logs/`: System and pipeline execution logs.

## 🎓 Conference Publications (ECCM22)

This repository serves as a companion to the research presented at the **22nd European Conference on Composite Materials (ECCM22)**. The following materials will be added to this folder:
- **Conference Paper**: The full research article as published in the ECCM22 proceedings.
- **Scientific Poster**: The visual presentation summarizing the study's methodology and results.

---
*Developed as part of the LZP-1-24-083 Post-Doc project.*
