from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from smiles.descriptor_builder import SafeDescriptorBuilder

app = FastAPI(title="PolymerLife QSPR API")

TG_MODEL_PATH = "models/saved/tg_xgb_stage1.pkl"
DUR_MODEL_PATH = "models/saved/durability_xgb_stage2.pkl"

if os.path.exists(TG_MODEL_PATH) and os.path.exists(DUR_MODEL_PATH):
    tg_model = joblib.load(TG_MODEL_PATH)
    dur_model = joblib.load(DUR_MODEL_PATH)
    builder = SafeDescriptorBuilder()
else:
    tg_model, dur_model, builder = None, None, None


class InferenceRequest(BaseModel):
    sample_name: str
    smiles: str
    t_env: float
    humidity: float
    exposure_hours: float


@app.post("/api/v1/predict")
async def make_prediction(req: InferenceRequest):
    if not tg_model:
        raise HTTPException(status_code=500, detail="The models are not loaded. Run training (main.py) first.")

    chem_features = builder.process_smiles(req.smiles)
    if chem_features is None:
        raise HTTPException(status_code=400, detail="SMILES parsing error.")

    col_names = builder.get_feature_names()
    X_chem = pd.DataFrame([chem_features], columns=col_names)

    # 2. Stage 1: Tg
    pred_tg = float(tg_model.predict(X_chem)[0])

    # 3. Stage 2: Долговечность
    X_full = X_chem.copy()
    X_full['Tg'] = pred_tg
    X_full['T_env'] = req.t_env
    X_full['humidity'] = req.humidity
    X_full['exposure_hours'] = req.exposure_hours

    pred_retention = float(dur_model.predict(X_full)[0])

    result = {
        "sample_name": req.sample_name,
        "prediction": {
            "predicted_Tg_K": round(pred_tg, 2),
            "predicted_retention_percent": round(pred_retention, 2)
        },
        "status": "success"
    }

    base_url = "http://ваш-ip-на-eosc:8000/api/v1/results/download"
    result['file_urls'] = {"lines_plot": f"{base_url}/friedman_lines.png",
                           "ea_plot": f"{base_url}/friedman_Ea_vs_alpha.png",
                           "data_csv": f"{base_url}/{result['sample_name']}_friedman_Ea.csv"
                           }

    return result
