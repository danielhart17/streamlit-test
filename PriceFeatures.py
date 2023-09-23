from pydantic import BaseModel

class PriceFeatures(BaseModel):
    open: float
    high: float
    low: float
    rsi: float
    zlema: float
    stochrsi: float
    percent_b: float
    qstick: float