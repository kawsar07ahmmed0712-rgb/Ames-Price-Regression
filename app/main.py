import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from House_Price.pipeline.deployment_predictor import DeploymentPredictionPipeline
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Ames Price Predictor",
    description="Render-safe Ames house price prediction app",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@lru_cache(maxsize=1)
def get_deployment_pipeline() -> DeploymentPredictionPipeline:
    return DeploymentPredictionPipeline(bundle_dir="artifacts/deployment")


def render_home(
    request: Request,
    predicted_price: Optional[str] = None,
    raw_price: Optional[float] = None,
    form_values: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> HTMLResponse:
    pipeline = get_deployment_pipeline()

    context = {
        "request": request,
        "options": pipeline.get_form_options(),
        "form_values": form_values or pipeline.get_default_form_values(),
        "predicted_price": predicted_price,
        "raw_price": raw_price,
        "error_message": error_message,
        "github_url": "https://github.com/kawsar07ahmmed0712-rgb/Ames-Price-Regression",
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context,
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_home(request=request)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": "Ames Price Predictor",
        "mode": "deployment-artifact",
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Neighborhood: str = Form(...),
    OverallQual: int = Form(...),
    GrLivArea: float = Form(...),
    GarageCars: int = Form(...),
    GarageArea: float = Form(...),
    YearBuilt: int = Form(...),
    YearRemodAdd: int = Form(...),
    TotalBsmtSF: float = Form(...),
    FirstFlrSF: float = Form(...),
    FullBath: int = Form(...),
    TotRmsAbvGrd: int = Form(...),
    ExterQual: str = Form(...),
    KitchenQual: str = Form(...),
    BsmtQual: str = Form(...),
    GarageFinish: str = Form(...),
):
    form_values = {
        "Neighborhood": Neighborhood,
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "TotalBsmtSF": TotalBsmtSF,
        "FirstFlrSF": FirstFlrSF,
        "FullBath": FullBath,
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "ExterQual": ExterQual,
        "KitchenQual": KitchenQual,
        "BsmtQual": BsmtQual,
        "GarageFinish": GarageFinish,
    }

    try:
        pipeline = get_deployment_pipeline()
        output = pipeline.predict(form_values)

        price = float(output["PredictedSalePrice"])

        return render_home(
            request=request,
            predicted_price=output["PredictedSalePriceFormatted"],
            raw_price=price,
            form_values=form_values,
            error_message=None,
        )

    except Exception as e:
        logger.error("Prediction request failed.")
        error_message = str(CustomException(e, sys))

        return render_home(
            request=request,
            predicted_price=None,
            raw_price=None,
            form_values=form_values,
            error_message=error_message,
        )
