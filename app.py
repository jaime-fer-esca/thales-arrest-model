from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ========================================
# CARGA DEL MODELO .JOBLIB
# ========================================
model = joblib.load("modelo.joblib")

app = FastAPI()

# ========================================
# ESTRUCTURA DE LOS FEATURES QUE TE MANDA N8N
# ========================================
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

    data = features.dict()

    # ------------ PREPROCESAMIENTO ------------
    # Aquí tú conviertes TEMP_CLASS, CLUSTER_TYPE, etc.
    # exactamente igual a como lo hiciste en tu pipeline
    #
    # EJEMPLO (luego lo rellenamos con tu lógica real):
    X = procesar_features(data)

    # ------------ PREDICCIÓN REAL ------------
    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)

    return {
        "ARREST_PRED": pred,
        "ARREST_PROBA": float(proba),
        "ARREST_SCORE": float(proba)
    }


# ========================================
# FUNCIÓN PROVISIONAL DE PREPROCESAMIENTO
# (LUEGO AQUI METEMOS TU PIPELINE REAL)
# ========================================
def procesar_features(data):
    # EJEMPLO: Lo vamos a reemplazar contigo después
    # Por ahora, esto evita que Render truene.
    arr = np.array([[0]*1])  
    return arr