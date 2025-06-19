from fastapi import FastAPI, UploadFile, File, Body
import pandas as pd
import json
from typing import Dict
import io
from fastapi.responses import JSONResponse, ORJSONResponse
from loguru import logger


app = FastAPI(default_response_class=ORJSONResponse)


def predict(df):
    df["prediction"] = 1
    return df


@app.post("/predict/")
async def handle_predict(file: UploadFile,
                   data = Body(None)):

    if file:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            return JSONResponse(status_code=400, content={"message": f"Error processing CSV: {str(e)}"})
        result_df = predict(df)
        return ORJSONResponse(content=result_df.to_dict(orient="records"))
    if data:
        logger.info("input data")
        logger.info("")
        logger.info("")
        logger.info(data)
        df = pd.DataFrame(json.loads(data))
        logger.info(df.head())
        result_df = predict(df)
        return ORJSONResponse(content=result_df.to_dict(orient="records"))
    return ORJSONResponse(status_code=400, content={"message": "Please provide either a file or structured data."})
