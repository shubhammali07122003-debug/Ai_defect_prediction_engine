from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="AI Defect Prediction Engine API",
    description="REST API for predicting code defect probability using ML models",
    version="1.0.0"
)

# Routes register karein
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
