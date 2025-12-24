# main.py
from fastapi import FastAPI, HTTPException
from friedman_analysis_air import run_friedman_analysis
from ozawa_analysis_air import run_ozawa_analysis
from kissinger_analysis_air import run_kissinger_analysis
from preprocessing import process_all_raw_files

app = FastAPI(
    title="TGA Kinetics Analysis API",
    description="Full pipeline: preprocessing + kinetic analysis (Friedman, Ozawa, Kissinger)",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to the TGA Kinetics API!",
        "endpoints": {
            "/preprocess": "Process source files from data_csv → data_modified",
            "/friedman": "Friedman's method (Eₐ(α))",
            "/ozawa": "Ozawa-Flynn-Wall method (Eₐ(α))",
            "/kissinger": "Kissinger method (average Eₐ)"
        }
    }


@app.get("/preprocess")
async def preprocess(
    input_dir: str = 'data_csv',
    output_dir: str = 'data_modified',
    window: int = 19
):
    """
    Preprocessing of all raw CSV files.
    """
    try:
        files = process_all_raw_files(
            input_dir=input_dir,
            output_dir=output_dir,
            window_length=window
        )
        return {
            "status": "success",
            "message": f"Fail {len(files)} - processed successfully",
            "processed_files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")


@app.get("/friedman")
async def friedman(
    input_dir: str = 'data_modified',
    output_dir: str = 'kinetics_results',
    alpha_start: float = 0.05,
    alpha_end: float = 0.96,
    alpha_step: float = 0.05,
    alpha_tolerance: float = 0.03
):
    """
    Friedman method with adjustable parameters α.
    """
    result = run_friedman_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_step=alpha_step,
        alpha_tolerance=alpha_tolerance
    )
    if result['status'] == 'error':
        raise HTTPException(status_code=500, detail=result['message'])
    return result


@app.get("/ozawa")
async def ozawa(
    input_dir: str = 'data_modified',
    output_dir: str = 'kinetics_results',
    alpha_start: float = 0.05,
    alpha_end: float = 0.96,
    alpha_step: float = 0.05,
    alpha_tolerance: float = 0.03
):
    """
    Ozawa-Flynn-Wall method with adjustable α parameters.
    """
    result = run_ozawa_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_step=alpha_step,
        alpha_tolerance=alpha_tolerance
    )
    if result['status'] == 'error':
        raise HTTPException(status_code=500, detail=result['message'])
    return result


@app.get("/kissinger")
async def kissinger(
    input_dir: str = 'data_modified',
    output_dir: str = 'kinetics_results',
    da_dt_column: str = 'dalpha_dt'
):
    """
    Kissinger method (finding the maximum for a specified column).
    """
    result = run_kissinger_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        da_dt_column=da_dt_column
    )
    if result['status'] == 'error':
        raise HTTPException(status_code=500, detail=result['message'])
    return result