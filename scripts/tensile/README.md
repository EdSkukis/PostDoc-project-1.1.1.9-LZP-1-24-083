# Batch Processor for Tensile Test Workbooks

This repository contains a single Python script for processing tensile test experiments.  
The script reads raw Excel workbooks, cleans data, computes key metrics, and produces reports.

## Features
- Load Excel workbooks (data sheet + metadata sheet)
- Clean unit/header rows and remove noise/spikes
- Compute:
  - Engineering stress–strain
  - Young’s modulus (linear fit on small strain window)
  - True stress–strain and Hollomon parameters (n, K)
  - 0.2% proof stress (Rp0.2) via the offset method
  - Maximum stress and failure time
- Plot annotated stress–strain curves (PNG)
- Generate combined CSV, Excel reports, and interactive Plotly HTML overview

## Usage
1. Place raw `.xls` / `.xlsx` files into the `data/` directory.
2. (Optional) Create a CSV file `data/overrides.csv` with the following structure to override test speed values from metadata:
   ```csv
   File,TestSpeed_mm_min
   sample1.xls,6.6
   sample2.xlsx,66
