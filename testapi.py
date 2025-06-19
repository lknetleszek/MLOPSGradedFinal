from loguru import logger
from ARISA_DSML.config import RAW_DATA_DIR
import pandas as pd
import requests
import json
from ARISA_DSML.config import RAW_DATA_DIR

url =  "http://localhost:8000/predict/"

if __name__=="__main__":
    mode = "file"  # or row

    logger.info(f"Testing {mode}")
    if mode=="row":
        df = pd.read_csv(RAW_DATA_DIR / "test.csv")
        data = df.head(1).to_dict()
        data = json.dumps(data)
        logger.info(data)
        response = requests.post(url, json=data)
        logger.info(response.content)
        df_pred = pd.DataFrame(json.loads(response.content))
        logger.info(df_pred)
    elif mode=="file":
        test_data = RAW_DATA_DIR / "new_test.csv"
        with test_data.open("rb") as f:
            response = requests.post(url, files={"file": f})
            logger.info(pd.DataFrame(response.json()).head())
