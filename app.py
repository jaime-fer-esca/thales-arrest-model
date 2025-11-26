from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# =====================================
#  Carga del modelo (pipeline completo)
# =====================================
# Asumimos que modelo.joblib es tu Pipeline:
# ColumnTransformer + modelo de clasificación
model = joblib.load("modelo.joblib")

app = FastAPI()

# Orden y nombres EXACTOS de las columnas
FEATURE_COLS = [
    "DATE_YEAR", "DATE_MONTH", "DATE_DAY", "DATE_HOUR",
    "LATITUDE", "LONGITUDE",
    "POPULATION_TOTAL", "OFFICER_TOTAL",
    "TEMP_CLASS", "GUNSHOT_INJURY",
    "CLUSTER_TYPE", "DOMESTIC",
    "DISTRICT_NAME", "UNAUTHORIZED_USE",
]


# ============================
#  Esquema de entrada (body)
# ============================
class Features(BaseModel):
    DATE_YEAR: int
    DATE_MONTH: int
    DATE_DAY: int
    DATE_HOUR: int
    LATITUDE: float
    LONGITUDE: float
    POPULATION_TOTAL: float
    OFFICER_TOTAL: float
    TEMP_CLASS: str
    GUNSHOT_INJURY: float
    CLUSTER_TYPE: str
    DOMESTIC: str
    DISTRICT_NAME: str
    UNAUTHORIZED_USE: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: Features):
    # 1) Body -> dict normal
    data = features.dict()

    # 2) Construimos un dict SOLO con las columnas que el modelo espera,
    #    en el orden correcto
    row = {col: data[col] for col in FEATURE_COLS}

    # 3) DataFrame de 1 fila x 14 columnas
    X = pd.DataFrame([row], columns=FEATURE_COLS)

    # 4) Predicción
    proba = float(model.predict_proba(X)[0][1])  # probabilidad clase 1
    pred = int(proba >= 0.5)                     # umbral 0.5 (ajustable)

    # 5) Respuesta para n8n
    return {
        "ARREST_PRED": pred,
        "ARREST_PROBA": proba,
        "ARREST_SCORE": proba,
    }
