# 3PB Batch Processor

A Python tool for **batch processing three-point bending (3PB)** experimental CSV files, automatically computing **stress–strain curves**, **Young’s modulus**, and key metrics, while generating publication-ready plots (PNG and interactive Plotly HTML).

---

## 1. Overview

The script automates post-processing of bending test data by:

- parsing CSVs with European (“comma-based”) decimals and quoted values;  
- reading geometry and E-window configuration from an `info.csv` file;  
- computing `stress–strain` curves from raw deformation–force data;  
- determining the **Young’s modulus** in a specified percentage window of the maximum force;  
- producing individual PNG plots, a combined PNG and Plotly HTML overlay;  
- generating a summary table with all derived metrics;  
- concatenating all data points into a single CSV file.

---

## 2. Installation

Requires **Python ≥ 3.9**.

```bash
pip install numpy pandas matplotlib plotly
