"""Prediction endpoints for single and batch inference.

Supports A/B testing when active, with shadow deployment
running transparently in the background.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_ab_manager, get_predictor
from src.api.schemas import PredictionRequest, PredictionResponse
from src.serving.ab_testing import ABTestManager
from src.serving.predictor import Predictor

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    predictor: Predictor = Depends(get_predictor),
    ab_manager: ABTestManager = Depends(get_ab_manager),
) -> PredictionResponse:
    """Run model prediction on input features.

    Supports single sample and batch predictions. If an A/B test is
    active for the requested model, traffic is automatically split
    between control and treatment variants.

    **Single prediction:**
    ```json
    {"features": [[5.1, 3.5, 1.4, 0.2]]}
    ```

    **Batch prediction:**
    ```json
    {"features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}
    ```
    """
    try:
        # Check for active A/B test
        test = ab_manager.get_test(request.model_id)

        if test is not None:
            ab_result = ab_manager.predict(
                features=request.features,
                model_id=request.model_id,
                request_id=request.request_id,
                include_probabilities=request.include_probabilities,
            )
            return PredictionResponse(
                predictions=ab_result.prediction.predictions,
                probabilities=ab_result.prediction.probabilities,
                model_id=ab_result.prediction.model_id,
                model_version=ab_result.prediction.model_version,
                latency_ms=ab_result.prediction.latency_ms,
                variant=ab_result.variant,
            )

        # Standard prediction
        result = predictor.predict(
            features=request.features,
            model_id=request.model_id,
            model_version=request.model_version,
            include_probabilities=request.include_probabilities,
        )

        return PredictionResponse(
            predictions=result.predictions,
            probabilities=result.probabilities,
            model_id=result.model_id,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e
