import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from House_Price.config.configuration import ConfigurationManager
from House_Price.pipeline.input_adapter import AmesRawInputAdapter
from House_Price.pipeline.prediction_pipeline import SinglePredictionPipeline
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Ames Price Predictor",
    description="Production ML app for Ames house price prediction",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@lru_cache(maxsize=1)
def get_configuration_manager() -> ConfigurationManager:
    return ConfigurationManager()


@lru_cache(maxsize=1)
def get_input_adapter() -> AmesRawInputAdapter:
    config_manager = get_configuration_manager()
    prediction_config = config_manager.get_prediction_config()

    raw_test_path = prediction_config.raw_test_path
    raw_train_path = str(Path(raw_test_path).with_name("train.csv"))

    return AmesRawInputAdapter(
        raw_train_path=raw_train_path,
        raw_test_path=raw_test_path,
        schema=prediction_config.schema,
    )


@lru_cache(maxsize=1)
def get_single_prediction_pipeline() -> SinglePredictionPipeline:
    config_manager = get_configuration_manager()
    prediction_config = config_manager.get_prediction_config()

    return SinglePredictionPipeline(config=prediction_config)


def get_default_form_values() -> Dict[str, Any]:
    return {
        "Neighborhood": "NridgHt",
        "OverallQual": 7,
        "GrLivArea": 1650,
        "GarageCars": 2,
        "GarageArea": 500,
        "YearBuilt": 2005,
        "YearRemodAdd": 2005,
        "TotalBsmtSF": 950,
        "FirstFlrSF": 950,
        "FullBath": 2,
        "TotRmsAbvGrd": 6,
        "ExterQual": "Gd",
        "KitchenQual": "Gd",
        "BsmtQual": "Gd",
        "GarageFinish": "RFn",
    }


def render_home(
    request: Request,
    predicted_price: Optional[str] = None,
    raw_price: Optional[float] = None,
    form_values: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> HTMLResponse:
    adapter = get_input_adapter()
    options = adapter.get_form_options()

    context = {
        "request": request,
        "options": options,
        "form_values": form_values or get_default_form_values(),
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
        adapter = get_input_adapter()
        raw_df = adapter.build_raw_dataframe(form_values)

        pipeline = get_single_prediction_pipeline()
        output = pipeline.predict_from_dataframe(
            raw_df=raw_df,
            strict_unknown_categories=False,
            save_latest=True,
        )

        price = float(output["predictions"][0]["PredictedSalePrice"])

        return render_home(
            request=request,
            predicted_price=f"${price:,.0f}",
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