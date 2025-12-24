# TGA Kinetics Analysis Project
A complete pipeline for the kinetic analysis of thermogravimetric (TGA) data
in air at different heating rates.
The project includes raw data preprocessing, calculation of the degree 
of conversion α, the rate dα/dt, and kinetic analysis using 
three isoconversion methods:
- Friedman (differential, model-free, Eₐ(α))
- Ozawa-Flynn-Wall (OFW) (integral, model-free, Eₐ(α))
- Kissinger (integral, single average Eₐ value)

The project is implemented as a FastAPI application with convenient endpoints 
for launching analysis via HTTP requests.

## Project structure
```
tga_kinetics/
├── data_csv/                       # ← Input fails. Data from experement (spacemen_name_XXmin_air.csv и т.д.)
├── data_modified/                  # ← Automatically created: processed files(*_processed.csv)
├── kinetics_results/               # ← Analysis results: CSV, PNG graphics
├── preprocessing
    ├── core.py            # Preprocessing of raw CSV (α, dα/dt, Savitzky-Golay smoothing)+
├── methods
    ├── friedman_analysis_air.py    # Friedman method (Eₐ(α))
    ├── ozawa_analysis_air.py       # Ozawa-Flynn-Wall (Eₐ(α)) method
    ├── kissinger_analysis_air.py   # Kissinger method (average Eₐ)
├── main.py                         # FastAPI application with endpoints
└── README.md
```
## Experimental Raw Data format
The source files in the folder data_csv/
Format: 0_sist_10min_air.csv, 0_sist_15min_air.csv...

The files must have the following structure:
```text
Index,Ts,Value
[C],[%]
0,42.3473,100
...
```
## Installation and start-up
### Requirements
- Python 3.9+
- Install dependencies:
```bash
pip install pandas numpy scipy matplotlib fastapi uvicorn python-multipart
```
## Usage
### Step 1: Start the API Server
Run the FastAPI app:
```bash
uvicorn main:app --reload
```
- Open http://127.0.0.1:8000 - welcome message
- Open http://127.0.0.1:8000/docs — interactive Swagger documentation with all parameters
### Step 2: Preprocess Raw Data 
This step processes raw TGA files to compute alpha, dalpha/dt, etc., and saves them in data_modified/.

API Call Example:
```
GET /preprocess?input_dir=data_csv&output_dir=data_modified&window=19
```
- Parameters:
    - `input_dir`: Folder with raw CSV (default: 'data_csv')
    - `output_dir`: Folder for processed CSV (default: 'data_modified')
    - `window`: Savitzky-Golay window length for smoothing (odd number, default: 19 for accurate dα/dt matching your examples)

Result: files in `output_dir/` with columns:
`time_s`, `T_C`, `T_K`, `inv_T_K`, `mass_percent`, `alpha_percent`, 
`dalpha_dt`, `ln_dalpha_dt`.

**Details:** The preprocessing uses Savitzky-Golay filter to smooth alpha_percent and compute gradient for dalpha/dt. It handles buoyancy effects by normalizing mass to [0, 100%].

### Step 3: Start kinetic analysis
After preprocessing, run the methods. Each endpoint accepts parameters for customization.
#### Friedman Method
Differential method: Plots ln(dα/dt) vs 1/T at fixed α levels.

API Call Example:
```
GET /friedman?alpha_start=0.05&alpha_end=0.96&alpha_step=0.05&alpha_tolerance=0.03&input_dir=data_modified
```
- Parameters (all optional):
    - input_dir: Folder with processed data (default: 'data_modified')
    - output_dir: Results folder (default: 'kinetics_results')
    - alpha_start: Start of alpha range (default: 0.05)
    - alpha_end: End of alpha range (default: 0.96)
    - alpha_step: Step for alpha levels (default: 0.05)
    - alpha_tolerance: Max deviation from target alpha (default: 0.03)

#### Ozawa-Flynn-Wall Method
Integral method: Plots ln(β) vs 1/T at fixed α.

API Call:
```
GET /ozawa?alpha_start=0.1&alpha_end=0.9&alpha_step=0.02&alpha_tolerance=0.05
```
Parameters same as Friedman (no dα/dt needed, only T at alpha).


#### Kissinger
Integral method for average Ea: Uses peak temperature T_p from dα/dt max.

API Call:
```
GET /kissinger?da_dt_column=dalpha_dt
```
- Parameters:
    - input_dir, output_dir (same as above)
    - da_dt_column: Column for rate to find peak (default: 'dalpha_dt')

## All methods save:
- CSV with Eₐ(α) or one value
- CSV with points for building lines (for verification)
- PNG-graphics (lines + Eₐ vs α)

## Troubleshooting
- No files found: Ensure raw CSV in data_csv/ match pattern *_XXmin_air.csv. 
- Insufficient curves: Need at least 2 heating rates for analysis. 
- NaN in Ea: Adjust alpha_tolerance if points not found. 
- Errors in smoothing: Window must be odd and < data length — adjust in preprocess. 
- Dependencies: If matplotlib fails, check installation.

## License
PostDoc-project-1.1.1.9-LZP-1-24-083 Free to use and modify.
Author: Eduards Skukis
Date: December 24, 2025
Version: 1.0.0