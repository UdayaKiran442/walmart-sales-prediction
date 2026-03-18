import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.utils.utils import split_date, load_object

class PredictSalesSchema(BaseModel):
    # Define the input schema for the predict_sales endpoint
    store: int
    dept: int
    date: str
    isHoliday: bool
    size: int

app = FastAPI()

@app.post("/predict-sales")
def predict_sales(payload: PredictSalesSchema):
    try:
        # convert payload to dataframe
        input_dict = {
            "Store": payload.store,
            "Dept": payload.dept,
            "IsHoliday": payload.isHoliday,
            "Size": payload.size,
            "Date": payload.date
        }
        input_df = pd.DataFrame([input_dict])
        # split date into year, month, day, weekofyear
        input_df = split_date(input_df)
        # drop date column
        input_df = input_df.drop("Date", axis=1)
        transformation_object = load_object('/Users/uday/Desktop/walmart-sales-prediction/Artifacts/data_transformation/transformated_object/preprocessor.pkl')
        input_df_transformed = transformation_object.transform(input_df)
        model = load_object('/Users/uday/Desktop/walmart-sales-prediction/Artifacts/model_trainer/trained_model.pkl')
        # make prediction
        prediction = model.predict(input_df_transformed)
        return {"message": "Prediction successful", "input_data": input_df_transformed.tolist(), "prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}