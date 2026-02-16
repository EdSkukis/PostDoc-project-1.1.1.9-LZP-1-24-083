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
    """Uploading one or more TXT files for analysis from TGA equipment
    File name have to consist of SpecimensName_xmin.txt
    where x is [5, 10, 15 or 20]
    EXAMPLE:
                                                                06.02.2026 15:48
________________________________________________________________________________



Curve Name:
  $!TGA_RG_PBAT (5C/min) N2
  Sample Weight
Curve Values:
          Index             Ts          Value
                          [�C]            [%]
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
    """Start cleaning and preparing all CSV files from data_csv"""
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
    """Running Friedman's method with dynamic links and file checking"""
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
    """Launch of the Ozawa Method(Ozawa-Flynn-Wall)"""
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
    """Launching the Kissinger method (based on heating rate peaks)"""
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
    """The results folder has been cleared. Files have been deleted."""

    def get_file_info(directory: Path):
        return [
            {
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 2),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
            for f in directory.glob("*") if f.is_file()
        ]

    return {
        "data_csv_raw": get_file_info(UPLOAD_DIR),
        "data_modified": get_file_info(Path("data_modified")),
        "kinetics_results": get_file_info(Path("kinetics_results"))
    }


@app.delete("/api/v1/storage/clear-uploads", tags=["Storage"])
async def clear_uploads():
    """Completely clearing the incoming CSV folder"""
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
    """Deletes all files from the results folder."""
    results_dir = Path("kinetics_results")
    deleted_count = 0
    for file in results_dir.glob("*"):
        if file.is_file():
            file.unlink()
            deleted_count += 1

    logger.info(f"The results folder has been cleared. Files have been deleted.: {deleted_count}")
    return {"status": "success", "message": f"Deleted {deleted_count} files."}


@app.get("/api/v1/results/download/{filename}", tags=["Storage"])
async def download_result(filename: str):
    """
    Safe download of results (PNG graphs or CSV tables).
    """
    results_dir = Path("/app/kinetics_results")

    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in results_dir.glob("*") if f.is_file()]

    if not files:
        logger.warning("Attempt to download empty results folder.")
        raise HTTPException(status_code=404, detail="No result files found. Run analysis first.")

    # Создаем ZIP
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                # Важно: используем str(file_path) для zip_file.write
                zip_file.write(str(file_path), arcname=file_path.name)

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=all_results.zip"}
        )
    except Exception as e:
        logger.error(f"Error during ZIP creation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal archiving error: {str(e)}")

@app.get("/api/v1/results/download-all", tags=["Storage"])
async def download_all_results():
    """
    Packs all files from the kinetics_results folder into a single ZIP archive and gives it to the user.
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


@app.get("/health")
async def health_check():
    """Checking the server status"""
    return {"status": "online", "storage_ok": os.path.exists("data_csv")}