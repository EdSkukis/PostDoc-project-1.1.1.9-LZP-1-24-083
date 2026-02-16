import os
import io
import logging
import shutil
import mimetypes
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import Request
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool
import zipfile
from datetime import datetime

from methods.fridman_analysis_air import run_friedman_analysis
from methods.kissinger_analysis_air import run_kissinger_analysis
from methods.ozawa_analysis_air import run_ozawa_analysis
from preprocessing.core import process_all_raw_files, logger as preproc_logger


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)
logger = logging.getLogger("API")

app = FastAPI(
    title="TGA Kinetics Cloud API",
    description="""
TGA data processing system:
- data preprocessing (file preparation for calculation)
- Friedman analysis
- Kissinger analysis
- Ozawa analysis
- Dowload analysis
""",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data_csv"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- Data model ---

class AnalysisConfig(BaseModel):
    alpha_start: float = Field(0.05, ge=0.01, description="Beginning of the alpha interval")
    alpha_end: float = Field(0.95, le=0.99, description="End of alpha interval")
    alpha_step: float = Field(0.05, gt=0)


def get_result_links(request: Request, sample_name: str, method_prefix: str) -> Dict:
    """
    Automatically checks for files on disk and generates URLs.
    method_prefix: например 'friedman' или 'ozawa'
    """
    host_url = f"{request.url.scheme}://{request.url.netloc}"
    base_download_url = f"{host_url}/api/v1/results/download"
    results_dir = Path("kinetics_results")

    # Словарь ожидаемых файлов для конкретного метода
    # Добавляем сюда логику именования, которую используют ваши .py скрипты
    expected_files = {
        "lines_plot": f"{sample_name}_{method_prefix}_lines.png",
        "ea_plot": f"{sample_name}_{method_prefix}_Ea_vs_alpha.png",
        "data_csv": f"{sample_name}_{method_prefix}_Ea.csv"
    }

    found_urls = {}
    missing = []

    for key, filename in expected_files.items():
        if (results_dir / filename).exists():
            found_urls[key] = f"{base_download_url}/{filename}"
        else:
            missing.append(filename)

    return {"urls": found_urls, "missing": missing}

# --- Endpoints ---

# ==== Data Preparation ===

@app.post("/api/v1/upload", tags=["Data Preparation"])
async def upload_tga_files(files: List[UploadFile] = File(...)):
    """
        ### Loading experimental data (TGA)

        This endpoint accepts files in **.txt** format received directly from the TGA equipment.

        ---
        ### File naming requirements:
        Files must follow the pattern: `{SampleName}_{HeatingRate}min.txt`

        * **SpecimensName**: The name of the material (eg `PBAT`).
        * **xmin**: Heating rate. Acceptable values: `5`, `10`, `15`, `20`.
        * **Example**: `PBAT_10min.txt`

        ---
        ### Expected structure inside the file:
        The parser searches for the data table after the header and stops at the first empty line.
       The following speakers are required:
        1.  **Index** — measurement number.
        2.  **Ts** — temperature (°C).
        3.  **Value** — mass (%).

        **Example of data format:**
        ```text
        Curve Values:
                  Index             Ts          Value
                                  [°C]            [%]
                      0        42.4621            100
                      5         42.829        100.032
        ```

        ---
        ### Validation:
        * Only files with the extension are accepted. `.txt`.
        * Files are saved to the `data_csv` folder for further processing..
        """
    uploaded_files = []
    errors = []

    for file in files:
        if not file.filename.endswith('.txt'):
            errors.append(f"File {file.filename} skipped: only CSV allowed")
            continue

        destination = UPLOAD_DIR / file.filename
        logger.info(f"Uploading file: {file.filename}")

        try:
            with destination.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)
        except Exception as e:
            logger.error(f"Error saving {file.filename}: {e}")
            errors.append(f"Could not save {file.filename}")
        finally:
            await file.close()

    return {
        "status": "success" if not errors else "partial_success",
        "uploaded": uploaded_files,
        "errors": errors,
        "message": f"Successfully uploaded {len(uploaded_files)} files."
    }


@app.post("/api/v1/preprocess", tags=["Data Preparation"])
async def preprocess_data():
    """
    ### Launching the data processing pipeline (Pipeline)

    This method performs a sequential transformation of raw data into a format ready for kinetic analysis.

    ---
    ### Processing stages:
    1.  **Convert TXT → CSV**: All `.txt` files from the `data_csv` folder are parsed with a search for the `Index, Ts, Value` headers.
    2.  **Filtering and Validation**:
        * Incorrect lines and gaps are removed.
        * Values are converted to numeric format.
        * Abnormal mass values are filtered out.
    3.  **Mathematical calculation**:
        * **T_K**: Temperature conversion from Celsius to Kelvin ($T + 273.15$).
        * **Alpha ($\alpha$)**: Calculates the degree of conversion from 0 to 1 based on the change in mass.
        * **dAlpha/dt**: Calculates the rate of change of the degree of conversion (differentiation with respect to time).
        * **ln(dAlpha/dt)**: Natural logarithm of the rate for subsequent Friedman analysis.

    ---
    ### Result:
    Processed files are saved in the `data_modified` folder with the suffix `_processed.csv`.
    These files become available for the **Friedman, Ozawa, and Kissinger** methods.

    ---
    ### Response format:
    * **converted_txt**: List of files successfully converted from TXT.
    * **processed_files**: Final list of files ready for analysis.
    """
    logger.info("Running a preprocessing task...")
    try:
        # Выполняем тяжелую задачу в отдельном потоке
        files = await run_in_threadpool(process_all_raw_files)
        if not files:
            logger.warning("Preprocessing completed, but no files were created.")
            return {"status": "warning", "message": "No files processed. Check input data."}

        logger.info(f"Files processed successfully: {len(files)}")
        return {"status": "success", "processed_files": files}
    except Exception as e:
        logger.error(f"Error in preprocessing endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==== Kinetics ===

@app.post("/api/v1/analyze/friedman", tags=["Kinetics"])
async def analyze_friedman(config: AnalysisConfig, request: Request):
    """
    ### Friedman's method (Differential Isoconversional Method)

    The Friedman method is used to determine the effective activation energy ($E_a$) of the thermal degradation process.
    It is based on the linear form of the Arrhenius equation for each degree of transformation ($\alpha$).

    ---
    ### What this method does:
    1. **Linearization**: For each selected degree of conversion ($\alpha$), a plot of $\ln(d\alpha/dt)$ versus $1/T$ is constructed.
    2. **Calculation of Ea**: Based on the slope of the obtained lines, the activation energy is calculated for each step in the interval.
    3. **Validation**: The coefficient of determination ($R^2$) is calculated for each line to assess the accuracy of the approximation.

    ---
    ### Results (Files):
    After a successful calculation, you will receive links to:
    * **{Sample}_friedman_lines.png**: Isoconversional Plots.
    * **{Sample}_friedman_Ea_vs_alpha.png**: Activation energy vs. conversion rate plot.
    * **{Sample}_friedman_Ea.csv**: Table with calculated $E_a$ values and statistical indicators.

    ---
    ### Requirements:
    * There must be at least three files with different heating rates (e.g., 5, 10, 20 °C/min) in the `data_modified` folder.
    * The files must go through the `/preprocess` step.

    ---
    ### Configuration parameters:
    * **alpha_start/end**: Analysis range (usually from 0.05 to 0.95).
    * **alpha_step**: Calculation step (e.g., 0.05). The smaller the step, the more detailed the graph, but the higher the load.
    """
    logger.info(f"Friedman launch for interval: {config.alpha_start}-{config.alpha_end}")

    # 1. Вызов тяжелого расчета в отдельном потоке
    result = await run_in_threadpool(
        run_friedman_analysis,
        alpha_start=config.alpha_start,
        alpha_end=config.alpha_end,
        alpha_step=config.alpha_step
    )

    # 2. Обработка внутренней ошибки метода
    if result.get('status') == 'error':
        logger.error(f"Friedman's method returned an error: {result.get('message')}")
        raise HTTPException(status_code=500, detail=result.get('message'))

    # 3. Проверка файлов и генерация ссылок
    sample_name = result.get('sample_name', 'result')
    links_info = get_result_links(request, sample_name, "friedman")

    # 4. Формирование финального ответа
    if links_info["missing"]:
        logger.warning(f"Friedman's calculation is complete, but the files are missing.: {links_info['missing']}")
        return {
            "status": "partial_success",
            "message": "The calculation was completed, but some result files were not found on the server..",
            "sample_name": sample_name,
            "missing_files": links_info["missing"],
            "file_urls": links_info["urls"]
        }

    logger.info(f"Friedman's method has been successfully completed for {sample_name}")
    return {
        "status": "success",
        "message": "The calculation was completed successfully. All files are available for download.",
        "sample_name": sample_name,
        "file_urls": links_info["urls"]
    }

@app.post("/api/v1/analyze/ozawa", tags=["Kinetics"])
async def analyze_ozawa(config: AnalysisConfig, request: Request):
    """
    ### Ozawa-Flynn-Wall (OFW) method

    The integral isoconversion method for determining activation energy ($E_a$).
    Unlike the Friedman method, the OFW method is less sensitive to experimental noise in the mass loss rate.

    ---
    ### Method Features:
    1. **Integral Approach**: The method is based on the dependence of temperature ($T$), corresponding to a fixed degree of conversion ($\alpha$), on the heating rate ($\beta$).
    2. **Linearization**: The dependence of $\log_{10}(\beta)$ on $1/T$ is plotted.
    3. **Equation**: The Doyle approximation is used, which allows one to obtain straight lines whose slope is directly proportional to the activation energy.
    ---
    ### Results (Files):
    * **ozawa_lines.png**: Plot of regression lines for different $\alpha$ values. Parallelism of these lines indicates that the reaction mechanism remains unchanged.
    * **ozawa_Ea_vs_alpha.png**: Plot of calculated activation energy versus reaction progress.
    * **{Sample}_ozawa_Ea.csv**: Full report with $E_a$ results, approximation errors, and $R^2$ coefficients.

    ---
    ### Data requirements:
    * A minimum of **3-4 curves** with significantly different heating rates (e.g., 5, 10, 15, 20 °C/min).
    * Files must be preprocessed through the `/preprocess` endpoint to convert to the $\alpha$ scale.
    ---
    ### Configuration:
    * **alpha_start/end**: Typically the interval $[0.1, 0.9]$ is analyzed to exclude the influence of initial and final fluctuations.
    """

    logger.info(f"Ozawa's launch for interval: {config.alpha_start}-{config.alpha_end}")

    # 1. Вызов расчета Озавы
    result = await run_in_threadpool(
        run_ozawa_analysis,
        alpha_start=config.alpha_start,
        alpha_end=config.alpha_end,
        alpha_step=config.alpha_step
    )

    if result.get('status') == 'error':
        logger.error(f"Ozawa's method returned an error: {result.get('message')}")
        raise HTTPException(status_code=500, detail=result.get('message'))

    # 2. Проверка файлов (используем префикс 'ozawa')
    sample_name = result.get('sample_name', 'result')
    links_info = get_result_links(request, sample_name, "ozawa")

    # 3. Формирование ответа
    return {
        "status": "success" if not links_info["missing"] else "partial_success",
        "message": result.get('message'),
        "sample_name": sample_name,
        "file_urls": links_info["urls"],
        "missing": links_info["missing"]
    }


@app.post("/api/v1/analyze/kissinger", tags=["Kinetics"])
async def analyze_kissinger(request: Request):
    """
    ### Kissinger Method

    A method for determining the activation energy ($E_a$) based on the shift in the temperature of the peak of
    the maximum reaction rate ($T_p$) when the heating rate ($\beta$) changes.
    ---
    ### Method Summary:
    1. **Peak Search**: For each thermogravimetric curve, the temperature $T_p$ at which the mass loss rate ($d\alpha/dt$)
    is maximum is determined.
    2. **Linearization**: A graph of $\ln(\beta / T_p^2)$ versus $1/T_p$ is plotted.
    3. **Single Result**: The slope of the resulting line allows one average value of the activation energy
    to be calculated for the entire degradation process.
    ---
    ### Results (Files):
    * **kissinger_lines.png**: Kissinger linear regression plot. The points on the plot correspond to the number
    of loaded files (experiments).
    * **{Sample}_kissinger_Ea.csv**: Summary table containing the $E_a$ value (kJ/mol), the pre-exponential factor ($A$),
    and the correlation coefficient $R^2$.
    ---
    ### Advantages and limitations:
    * **Advantage**: Does not require pre-setting of $\alpha$ intervals, works extremely quickly.
    * **Limitation**: Provides only one averaged $E_a$ value, not taking into account the change in
    the reaction mechanism during the transformation (unlike Friedman).
    ---
    ### Important:
    To correctly plot the line, you must use data from at least **3 different experiments** with different heating rates.
    """

    logger.info("Launching the Kissinger Method")

    # 1. Вызов расчета Киссинджера
    result = await run_in_threadpool(run_kissinger_analysis)

    if result.get('status') == 'error':
        logger.error(f"Kissinger's method returned an error.: {result.get('message')}")
        raise HTTPException(status_code=500, detail=result.get('message'))

    sample_name = result.get('sample_name', 'result')
    expected_files = {
        "kissinger_plot": f"{sample_name}_kissinger_lines.png",
        "data_csv": f"{sample_name}_kissinger_Ea.csv"
    }

    host_url = f"{request.url.scheme}://{request.url.netloc}"
    base_url = f"{host_url}/api/v1/results/download"
    results_dir = Path("kinetics_results")

    file_urls = {}
    missing = []

    for key, fname in expected_files.items():
        if (results_dir / fname).exists():
            file_urls[key] = f"{base_url}/{fname}"
        else:
            missing.append(fname)
    if missing:
        logger.warning(f"Kissinger: missing files {missing}")
        return {
            "status": "partial_success",
            "message": "The calculation is complete, but not all result files have been created..",
            "missing": missing,
            "file_urls": file_urls
        }

    logger.info(f"Kissinger's method has been successfully completed for{sample_name}")
    return {
        "status": "success",
        "message": "The Kissinger method was successfully completed. The graph and data are available.",
        "sample_name": sample_name,
        "file_urls": file_urls
    }


# ==== Storage ===


@app.get("/api/v1/storage/files", tags=["Storage"])
async def list_files():
    """
    ### File Storage Browser (Explorer)

    This method allows you to view the contents of all working directories on the server.
    It returns detailed information about files, including their size and last modification time.
    ---
    ### Folder structure:
    1. **data_csv_raw**: Original files (.txt or .csv) uploaded by the user.
    2. **data_modified**: Files that have passed the `/preprocess` stage. Contain the calculated $\alpha$ and $T_K$ parameters.
    3. **kinetics_results**: Final graphs (PNG) and tables (CSV) after the Friedman, Ozawa, or Kissinger analysis.
    ---
    ### Returned data:
    A JSON object is returned for each file:
    * `name`: Full file name.
    * `size_kb`: Size in kilobytes.
    * `modified`: Date and time of last save (YYYY-MM-DD HH:MM:SS).

    ---
    ### Useful for:
    * Checking whether files have been successfully uploaded.
    * Checking for the availability of processed data before running calculations.
    * Monitoring generated reports.
    """

    def _get_file_info(directory: Path):
        return [
            {
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 2),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
            for f in directory.glob("*") if f.is_file()
        ]

    return {
        "data_csv_raw": _get_file_info(UPLOAD_DIR),
        "data_modified": _get_file_info(Path("data_modified")),
        "kinetics_results": _get_file_info(Path("kinetics_results"))
    }


@app.get("/api/v1/results/download/{filename}", tags=["Storage"])
async def download_result(filename: str):
    """
    ### Downloading a specific result file

    Allows you to get a single graph (PNG) or table (CSV) by its name.

    ---
    ### Security:
    Filename validation is used to prevent path traversal attacks.

    ---
    ### How it works:
    1. The system searches for the file in the `kinetics_results` folder.
    2. Automatically detects the MIME type (the image will open in the browser, the table will download).
    3. If the file is not found, a 404 error is returned.
    """
    # Защита: берем только имя файла, игнорируя пути
    safe_filename = Path(filename).name
    file_path = Path("kinetics_results") / safe_filename

    if not file_path.exists() or not file_path.is_file():
        logger.warning(f"Запрос несуществующего файла: {safe_filename}")
        raise HTTPException(status_code=404, detail="Файл не найден.")

    # Определение типа контента (image/png, text/csv и т.д.)
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = 'application/octet-stream'

    return FileResponse(
        path=file_path,
        media_type=content_type,
        filename=safe_filename
    )


@app.get("/api/v1/results/download-all", tags=["Storage"])
async def download_all_results():
    """
    ### Download the full project archive (ZIP)

    The method automatically collects analysis results and preprocessed data into a single structured archive.
    ---
    ### Archive contents:
    The files within the archive are organized into folders for convenience:
    * **results/**: All generated graphs (PNG) and tables (CSV) for the Friedman, Ozawa, and Kissinger methods.
    * **processed_data/**: All CSV files obtained after the /preprocess step (containing $\alpha$, $T_K$, and destruction rates).

    ---
    ### Technical details:
    1. **In-memory compression**: The ZIP file is created on the fly in the server's RAM, eliminating the accumulation of temporary files on disk.
    2. **Streaming**: Transfer begins immediately after compression, allowing for efficient delivery of even large data sets.
    3. **Error Handling**: If there are no files after clearing storage or before starting calculations, the server will return a 404 error with an explanation.
    """
    folders_to_include = {
        "results": Path("kinetics_results"),
        "processed_data": Path("data_modified")
    }

    zip_buffer = io.BytesIO()
    files_found = 0

    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for folder_name, folder_path in folders_to_include.items():
                if folder_path.exists():
                    for file_path in folder_path.glob("*"):
                        if file_path.is_file():
                            # Сохраняем структуру внутри архива: folder/filename
                            zip_file.write(str(file_path), arcname=f"{folder_name}/{file_path.name}")
                            files_found += 1

        if files_found == 0:
            logger.warning("Attempt to download an empty archive.")
            raise HTTPException(status_code=404, detail="No files found to archive.")

        zip_buffer.seek(0)

        logger.info(f"A common archive has been created: {files_found} files from results and processing.")

        return StreamingResponse(
            zip_buffer,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=tga_full_results.zip"}
        )
    except Exception as e:
        logger.error(f"Error creating full archive: {e}")
        raise HTTPException(status_code=500, detail=f"Internal archiving error: {str(e)}")


@app.delete("/api/v1/storage/clear-uploads", tags=["Storage"])
async def clear_uploads():
    """
    ### Completely clear the uploads area (Reset)

   This method deletes all downloaded files and intermediate processing results. Use it before starting
   a new project or series of experiments.
    ---
    ### What will be deleted:
    1. **data_csv/**: All original `.txt` and `.csv` files uploaded via `/upload`.
    2. **data_modified/**: All files with the `_processed.csv` suffix created during the preprocessing stage.
    ---
    ### Important note:
    * This method does not affect the `kinetics_results` folder (the final graphs and reports remain).
    * Deletion is irreversible. Make sure you have downloaded all necessary data before calling this method.

    ---
    ### Server response:
    * `status`: Confirmation of successful deletion.
    * `deleted_files`: Total number of files deleted from both directories.
    """
    deleted_count = 0
    # Очищаем и сырые данные, и промежуточные (modified)
    for folder in [UPLOAD_DIR, Path("data_modified")]:
        for file in folder.glob("*"):
            if file.is_file():
                file.unlink()
                deleted_count += 1

    logger.info(f"Incoming data deleted. Files: {deleted_count}")
    return {"status": "success", "deleted_files": deleted_count}


@app.delete("/api/v1/results/clear", tags=["Storage"])
async def clear_results():
    """
    ### Final Results Cleanup

    The method permanently deletes all generated graphs and summary tables from the `kinetics_results` folder.
    ---
    ### What will be removed:
    * All PNG graphs (Friedman, Ozawa, Kissinger).
    * All CSV files with activation energy calculation results.

    ---
    ### Important note:
    * This method does not delete the original data in `data_csv` and `data_modified`.
    * It is recommended to call this method after downloading the archive using `/download-all`.
    ---
    ### Important note:
    * This method does not delete the original data in `data_csv` and `data_modified`.
    * It is recommended to call this method after downloading the archive using `/download-all`.
    """

    results_dir = Path("kinetics_results")
    deleted_count = 0
    for file in results_dir.glob("*"):
        if file.is_file():
            file.unlink()
            deleted_count += 1

    logger.info(f"The results folder has been cleared. Files have been deleted.: {deleted_count}")
    return {"status": "success", "message": f"Deleted {deleted_count} files."}


@app.get("/health")
async def health_check():
    """
    ### Server Health Check

    A method for monitoring API availability and checking file system health. Used for automated diagnostics.

    ---
    ### Parameters checked:
    1. **Status**: If 'online' is returned, the ASGI server is successfully processing requests.
    2. **Storage**: Checks whether the root folder for uploads exists. If 'false', there may be issues with
    access rights or mounting volumes in Docker.
    3. **Uptime Context**: Ensures that the server is not only running, but also ready to write data.
    ---
    ### Response format:
    * `status`: Current status (online).
    * `storage_ok`: Boolean value indicating access to disk storage.
    """
    return {"status": "online", "storage_ok": os.path.exists("data_csv")}