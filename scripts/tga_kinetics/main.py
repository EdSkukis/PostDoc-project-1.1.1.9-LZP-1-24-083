import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

# Импорты вашей структуры
from methods.fridman_analysis_air import run_friedman_analysis
from methods.kissinger_analysis_air import run_kissinger_analysis
from methods.ozawa_analysis_air import run_ozawa_analysis
from preprocessing.core import process_all_raw_files, logger as preproc_logger

# Настройка системного лога API
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)
logger = logging.getLogger("API")

app = FastAPI(
    title="TGA Kinetics Cloud API",
    description="Система обработки ТГА данных для EOSC",
    version="1.0.0"
)


# --- Модели данных ---
class AnalysisConfig(BaseModel):
    alpha_start: float = Field(0.05, ge=0.01, description="Начало интервала альфа")
    alpha_end: float = Field(0.95, le=0.99, description="Конец интервала альфа")
    alpha_step: float = Field(0.05, gt=0)


# --- Эндпоинты ---

@app.post("/api/v1/preprocess", tags=["Data Preparation"])
async def preprocess_data():
    """Запуск очистки и подготовки всех CSV файлов из data_csv"""
    logger.info("Запуск задачи предобработки...")
    try:
        # Выполняем тяжелую задачу в отдельном потоке
        files = await run_in_threadpool(process_all_raw_files)
        if not files:
            logger.warning("Предобработка завершена, но файлы не созданы.")
            return {"status": "warning", "message": "No files processed. Check input data."}

        logger.info(f"Успешно обработано файлов: {len(files)}")
        return {"status": "success", "processed_files": files}
    except Exception as e:
        logger.error(f"Ошибка в эндпоинте предобработки: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/friedman", tags=["Kinetics"])
async def analyze_friedman(config: AnalysisConfig):
    """Расчет методом Фридмана"""
    logger.info(f"Запуск Фридмана: alpha {config.alpha_start}-{config.alpha_end}")

    result = await run_in_threadpool(
        run_friedman_analysis,
        alpha_start=config.alpha_start,
        alpha_end=config.alpha_end,
        alpha_step=config.alpha_step
    )

    if result.get('status') == 'error':
        logger.error(f"Ошибка метода Фридмана: {result.get('message')}")
        raise HTTPException(status_code=500, detail=result.get('message'))

    return result


@app.get("/api/v1/results/download/{filename}", tags=["Storage"])
async def download_result(filename: str):
    """Скачивание графиков или CSV результатов"""
    file_path = Path("kinetics_results") / filename
    if not file_path.exists():
        logger.warning(f"Запрос несуществующего файла: {filename}")
        raise HTTPException(status_code=404, detail="Файл не найден")

    return FileResponse(path=file_path, filename=filename)


@app.get("/health")
async def health_check():
    """Проверка состояния сервера для мониторинга EOSC"""
    return {"status": "online", "storage_ok": os.path.exists("data_csv")}