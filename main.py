from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

# ======================
# مدل و تنظیمات
# ======================
class PredictionRequest(BaseModel):
    values: List[float]

model = None
features_count = None

# ======================
# بارگذاری مدل
# ======================
def load_model():
    global model, features_count

    csv_path = os.path.join(os.path.dirname(__file__), "DATA.CSV")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"DATA.CSV پیدا نشد: {csv_path}")

    df = pd.read_csv(csv_path, skiprows=[0])

    df.columns = [
        'Time(ms)', 'HeartRate', 'MicValue',
        'AmbientTemp', 'ObjectTemp', 'bidary'
    ]

    df = df.dropna()

    for col in ['Time(ms)', 'HeartRate', 'MicValue', 'AmbientTemp', 'ObjectTemp']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def time_to_min(val):
        try:
            if pd.isna(val):
                return np.nan
            if ':' in str(val):
                h, m = str(val).split(":")
                return int(h) * 60 + int(m)
            return np.nan
        except:
            return np.nan

    df['bidary'] = df['bidary'].apply(time_to_min)

    df = df.dropna()

    if len(df) < 1:
        raise ValueError("دیتاست کافی نیست")

    X = df[['Time(ms)', 'HeartRate', 'MicValue', 'AmbientTemp', 'ObjectTemp']].values
    y = df['bidary'].values

    features_count = X.shape[1]

    model = LinearRegression()
    model.fit(X, y)

    print("✅ مدل آماده شد")
    print("📊 تعداد ویژگی‌ها:", features_count)

# ======================
# Lifespan
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting...")
    load_model()
    yield
    print("🛑 Stopping...")

# ======================
# App
# ======================
app = FastAPI(
    title="Sleep Prediction API",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Routes
# ======================
@app.get("/")
def root():
    return {
        "status": "ok",
        "features_count": features_count
    }

@app.get("/health")
def health():
    return {
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(req: PredictionRequest):
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    if len(req.values) != features_count:
        raise HTTPException(
            status_code=400,
            detail=f"باید {features_count} عدد بدهی"
        )

    try:
        input_data = np.array([req.values])
        result = float(model.predict(input_data)[0])

        result = max(1, int(result))

        h = result // 60
        m = result % 60

        return {
            "success": True,
            "minutes": result,
            "display": f"{h}h {m}m"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# LOCAL RUN ONLY
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)