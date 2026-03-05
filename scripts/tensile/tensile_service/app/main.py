import uuid, shutil, os, pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, Cookie, Response, Depends, Query, HTTPException, Request
from fastapi.responses import FileResponse

# Импорт внутренних модулей для обработки и физики
from preprocessing.parser import split_raw_txt_by_test
from preprocessing.cleaning import clean_tensile_data
from core.physics import calculate_youngs_modulus, calculate_rp_offset
from core.analysis import detect_outliers
from core.visualization import plot_individual_test, plot_combined_tests

app = FastAPI(
    title="Tensile Data Processing System (TDPS)",
    version="1.1.2",
    description="API for automated mechanical properties characterization and statistical analysis."
)

STORAGE = Path("./storage")
STORAGE.mkdir(exist_ok=True)


def get_uid(user_id: Optional[str] = Cookie(None)):
    """Automatic generation or retrieval of user session ID from Cookies."""
    return user_id if user_id else str(uuid.uuid4())


@app.get("/", tags=["General"], summary="Welcome to Tensile Analysis API")
async def root():
    return {
        "status": "online",
        "message": "Welcome to Tensile Analysis API",
        "docs": "/docs"
    }

@app.get("/guide", tags=["General"], summary="Methodological manual on analysis parameters")
async def get_user_guide():
    """
   Returns a detailed description of the physical parameters used in
   calculations and recommendations for their configuration.
    """
    return {
        "title": "Tensile Data Processing System (TDPS) Methodological Guide",
        "parameters": {
            "E_modulus": {
                "name": "E-modulus",
                "unit": "MPa",
                "description": "A measure of material stiffness. Calculated as the slope of the curve in the elastic zone.",
                "setup_tips": "Recommended range of e_start_pct: 0.05-0.1% (allowance for gripper play). "
                              "e_end_pct: up to 0.8% (before the start of plastic surgery)."
            },
            "Rp_offset": {
                "name": "Conventional yield strength (Rp)",
                "unit": "MPa / %",
                "description": "The stress at which the residual strain is a specified percentage (usually 0.2%).",
                "standard": "ISO 6892-1 / ASTM E8"
            },
            "UTS": {
                "name": "Ultimate Tensile Strength - UTS",
                "unit": "MPa",
                "description": "The maximum stress a material can withstand before failure."
            },
            "Break_Threshold": {
                "name": "Threshold of destruction",
                "unit": "% from UTS",
                "description": "Criterion for automatic data trimming after sample destruction.",
                "action": "If there's excess data in the report, increase the threshold. If the graph is cut off too early, decrease it."
            }
        },
        "statistical_notes": {
            "Outliers": "Outliers are determined using the Z-score method (> 2.0 standard deviations).",
            "Mean_Calculation": "Means (MEAN) and standard deviations (STD) are calculated only from 'clean' data, excluding outliers."
        }
    }

@app.get("/files",
         tags=["File Management"],
         summary="List of files with links",
         description="Returns list of files with full URls that can be open/download in a browser")
async def list_files(request: Request, uid: str = Depends(get_uid)):
    """
    Automatically finds the current user's files using cookies.
    No UID is required. Links are generated through the secure /my-static/ endpoint.
    """
    user_dir = STORAGE / uid

    if not user_dir.exists():
        return {
            "message": "You don't have any uploaded data or results yet.",
            "user_id": uid,
            "files": []
        }

    base_url = str(request.base_url).rstrip("/")
    files_links = []

    # Рекурсивно сканируем папку пользователя (результаты и графики)
    for root_path, dirs, files in os.walk(user_dir):
        # Пропускаем папку с исходными сырыми данными, если это нужно
        if "raw" in root_path:
            continue

        for file in files:
            # Скрываем системные файлы
            if file.startswith('.') or file.endswith('.zip'):
                continue

            # Относительный путь (например, 'plots/test_0.png')
            rel_path = os.path.relpath(os.path.join(root_path, file), user_dir)
            clean_rel_path = rel_path.replace(os.sep, '/')

            # Формируем красивую прямую ссылку БЕЗ UID в пути
            full_url = f"{base_url}/my-static/{clean_rel_path}"

            files_links.append({
                "name": file,
                "folder": os.path.basename(root_path),
                "url": full_url
            })

    return {
        "user_id": uid,
        "total_files": len(files_links),
        "links": files_links
    }


@app.get("/my-static/{file_path:path}",
         tags=["File Management"],
         summary="Open/Download a file",
         description="Allows you to open an image or download an Excel file via a direct link. Example: http://62.3.175.167:8080/static/plots/test_0.png")
async def serve_my_file(file_path: str, uid: str = Depends(get_uid)):
    """
    Secure endpoint: serves a file only if it belongs to a user with the UID in the current Cookies.
    """
    # lstrip("/") предотвращает попытки выйти за пределы папки через ../../
    safe_file_path = file_path.lstrip("/")
    full_path = STORAGE / uid / safe_file_path

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found or access denied")

    return FileResponse(full_path)


@app.post("/upload",
          tags=["Data Processing"],
          summary="Uploading a TXT file",
          description="""
Accepts a TXT file with test data. The file must contain data blocks with the following header:

    Strain (%),Stress (MPa)
Example structure:
            
    0.138,-0.00825
    0.138,-0.0259
**Attention:** With each new download, all previous user data and results are deleted!
          """)
async def upload(
        response: Response,
        file: UploadFile = File(..., description="File with columns: Strain (%), Stress (MPa)"),
        encoding: str = Query("utf-8", description="File encoding (utf-8, windows-1251,...)"),
        uid: str = Depends(get_uid)
):
    """
    Downloads the file and clears the previous user session.
    Important: Old results are deleted with each new download!
    """
    response.set_cookie(key="user_id", value=uid, max_age=2592000, httponly=True)
    u_dir = STORAGE / uid
    if u_dir.exists(): shutil.rmtree(u_dir)
    raw_dir = u_dir / "raw";
    raw_dir.mkdir(parents=True)

    file_path = raw_dir / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        tests = split_raw_txt_by_test(str(file_path), encoding=encoding)
        return {"status": "Success", "found_experiments": len(tests), "message": "File uploaded successfully"}
    except Exception as e:
        shutil.rmtree(u_dir)
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")


@app.post("/process",
          tags=["Data Processing"],
          summary="Starting the analysis (Calculations)",
          description="""
### Physical and mechanical analysis of tensile data

This endpoint performs complex processing of experimental data to determine 
key material characteristics in accordance with international standards (ISO 6892-1, ASTM E8).
#### Calculated parameters:

1. **Young's Modulus (E, Young's Modulus):**
Defines the stiffness of the material. Calculated as the slope of the linear portion of the stress-strain 
curve (sigma/epsilon) using linear regression over a given strain range.
2. **Yield Strength (Rp, Yield Strength):**
The stress at which a given plastic strain occurs (usually 0.2%). Calculated by intersecting the curve with 
a line parallel to Young's Modulus and offset by the value rp_offset.
3. **Ultimate Tensile Strength (UTS, Ultimate Tensile Strength):**
The maximum stress a specimen can withstand before failure or necking.
4. **Strain at Break (epsilon_break, Elongation at Break):**
The maximum elongation of the specimen at the moment of physical separation.
5. **Break Threshold:**
A scientific criterion for cutting off "post-experimental noise." It defines the test termination 
point when the stress drops below a specified % of UTS due to structural failure.

#### Output data:
* **Individual graphs:** Visualization of calculated points, projections onto axes, and fitting lines.
* **Excel summary report:** A complete results table with automatic statistical processing.
* **Outlier analysis:** Identification of anomalous tests based on the Z-score (2sigma) method.
          """)
async def process(
        save_plots: bool = Query(True, description="Generate and save PNG graphics?"),
        mode: str = Query("strain", description="Window mode: 'strain' or 'stress'"),
        e_start_pct: float = Query(0.1, description="Beginning of the linear section for Young's modulus (in % strain)"),
        e_end_pct: float = Query(1.5, description="End of the linear section for Young's modulus (in % strain)"),
        rp_offset_pct: float = Query(0.2, description="Offset for Rp calculation (usually 0.2%)"),
        break_threshold_pct: float = Query(
            15.0,
            description="Destruction threshold (% of UTS). If the voltage drops below this level after the peak, the data is truncated."
        ),
        uid: str = Depends(get_uid)
):
    """
    MAIN BRAIN:
    1. Converts input percentages (%) to fractions (0.01) for physical formulas.
    2. Applies 'Smart Cut' based on the break_threshold_pct threshold.
    3. Calculates Young's modulus (E), yield strength (Rp), UTS, and strain at break.
    4. Generates an Excel summary report with automatic outlier filtering.
    """
    user_dir = STORAGE / uid
    raw_file = next((user_dir / "raw").glob("*"), None)
    if not raw_file:
        raise HTTPException(status_code=400, detail="Please upload the file first!")

    # КОРРЕКЦИЯ ВВОДА: Перевод процентов в доли для физического движка
    e_window = (e_start_pct / 100.0, e_end_pct / 100.0)
    abs_offset = rp_offset_pct / 100.0

    res_dir = user_dir / "results"
    plt_dir = user_dir / "plots"
    res_dir.mkdir(exist_ok=True)
    if save_plots:
        plt_dir.mkdir(exist_ok=True)

    test_blocks = split_raw_txt_by_test(str(raw_file))
    summary_data = []
    bundle_for_plots = []
    excel_path = res_dir / "report_full.xlsx"

    # 1. Основной цикл обработки тестов
    # Сначала собираем все данные, записывать в Excel будем в конце (так надежнее на Mac)
    all_sheets = {}

    for i, df_raw in enumerate(test_blocks):
        # Очистка данных с учетом пользовательского порога обрезания "хвоста"
        df = clean_tensile_data(df_raw, break_threshold=break_threshold_pct)

        # Расчет модуля Юнга (E)
        E, intercept, e_pts = calculate_youngs_modulus(
            df['strain_abs'].values, df['stress_mpa'].values, e_window, mode=mode
        )

        # Расчет Rp (Yield Strength)
        rp_stress, rp_strain = calculate_rp_offset(
            df['strain_abs'].values, df['stress_mpa'].values, E, abs_offset
        )

        # Максимальное напряжение (UTS)
        idx_max = df['stress_mpa'].idxmax()
        uts_stress = float(df.loc[idx_max, 'stress_mpa'])
        uts_strain_pct = float(df.loc[idx_max, 'strain_pct'])

        # Максимальное удлинение (Strain at Break)
        # Это последняя точка ПОСЛЕ применения break_threshold_pct
        break_strain_pct = float(df['strain_pct'].iloc[-1])

        # Визуализация индивидуального теста
        if save_plots and E:
            plot_individual_test(df, plt_dir / f"test_{i}.png", E, intercept, e_pts, rp_stress, rp_offset_pct, i)

        # Сохраняем "сырые" данные для Excel
        all_sheets[f"Test_{i}"] = df

        # Сбор данных для сводной таблицы
        summary_data.append({
            "id": i,
            "E_MPa": round(E, 2) if E else None,
            "Rp_MPa": round(rp_stress, 2) if rp_stress else None,
            "Rp_strain_%": round(rp_strain * 100.0, 4) if rp_strain else None,
            "UTS_MPa": round(uts_stress, 2),
            "UTS_strain_%": round(uts_strain_pct, 4),
            "Break_strain_%": round(break_strain_pct, 4),
            "is_outlier": False
        })
        bundle_for_plots.append({'df': df, 'E': E, 'intercept': intercept, 'e_pts': e_pts, 'rp': rp_stress})

    # 2. Поиск выбросов (Outliers) по Z-score
    final_summary = detect_outliers(summary_data, threshold=2.0)
    df_summary = pd.DataFrame(final_summary)

    # 3. Расчет расширенной статистики (Mean/Std) без учета выбросов
    clean_stats_df = df_summary[df_summary['is_outlier'] == False]
    if not clean_stats_df.empty:
        # Список колонок, для которых считаем среднее
        cols_to_stat = ['E_MPa', 'Rp_MPa', 'Rp_strain_%', 'UTS_MPa', 'UTS_strain_%', 'Break_strain_%']

        # Считаем среднее и стандартное отклонение
        mean_row = clean_stats_df[cols_to_stat].mean().round(2).to_dict()
        mean_row['id'] = 'MEAN (clean)'

        std_row = clean_stats_df[cols_to_stat].std().round(2).to_dict()
        std_row['id'] = 'STD (clean)'

        # Добавляем строки статистики в низ таблицы
        df_summary = pd.concat([df_summary, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # 4. Финальная запись в Excel (одним открытием файла для избежания таймаутов)
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Лист со сводными результатами - первый
        df_summary.to_excel(writer, sheet_name="Summary_Results", index=False)
        # Листы с данными каждого теста
        for sheet_name, data_df in all_sheets.items():
            data_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # 5. Построение общего сравнительного графика
    if save_plots and bundle_for_plots:
        plot_combined_tests(bundle_for_plots, user_dir / "combined_analysis.png", rp_offset_pct)

    return {"status": "Success", "results": final_summary}


@app.get("/download",
         tags=["Data Processing"],
         summary="Download the ZIP archive of results",
         description="Collects all files from the current session (raw data, PNG graphs, and Excel reports) into "
                     "a single ZIP archive and delivers it to the user. After downloading, the data remains on "
                     "the server until the next upload.")
async def download(uid: str = Depends(get_uid)):
    """Collects all charts and Excel files into one archive."""
    user_dir = STORAGE / uid
    if not user_dir.exists():
        raise HTTPException(status_code=404, detail="No data available")

    archive = STORAGE / f"results_{uid}"
    shutil.make_archive(str(archive), 'zip', str(user_dir))
    return FileResponse(path=f"{archive}.zip", filename="tensile_results.zip")