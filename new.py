from typing import Union
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

class Item(BaseModel):
    LIMIT_BAL: int
    SEX: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1:int
    BILL_AMT2:int
    BILL_AMT3:int
    BILL_AMT4:int
    BILL_AMT5:int
    BILL_AMT6:int
    PAY_AMT1:int
    PAY_AMT2:int
    PAY_AMT3:int
    PAY_AMT4:int
    PAY_AMT5:int
    PAY_AMT6:int


app = FastAPI()


@app.post("/items/")
async def create_item(item: Item):
    data={
  "LIMIT_BAL": item.LIMIT_BAL,
  "SEX": item.SEX,
  "AGE": item.AGE,
  "PAY_0": item.PAY_0,
  "PAY_2": item.PAY_2,
  "PAY_3": item.PAY_3,
  "PAY_4": item.PAY_4,
  "PAY_5": item.PAY_5,
  "PAY_6": item.PAY_6,
  "BILL_AMT1": item.BILL_AMT1,
  "BILL_AMT2": item.BILL_AMT2,
  "BILL_AMT3": item.BILL_AMT3,
  "BILL_AMT4": item.BILL_AMT4,
  "BILL_AMT5": item.BILL_AMT5,
  "BILL_AMT6": item.BILL_AMT6,
  "PAY_AMT1": item.PAY_AMT1,
  "PAY_AMT2": item.PAY_AMT2,
  "PAY_AMT3": item.PAY_AMT3,
  "PAY_AMT4": item.PAY_AMT4,
  "PAY_AMT5": item.PAY_AMT5,
  "PAY_AMT6": item.PAY_AMT6
}
    loaded_model = pickle.load(open('RFClassifier.pkl', 'rb'))
    d = pd.read_csv(r'UCI_Credit_Card.csv')
    data_in=pd.DataFrame([data])
    df =d.copy()
    df = df.drop("ID",axis=1)
    df = df.drop("EDUCATION",axis=1)
    df = df.drop("MARRIAGE",axis=1)
    X= df.drop('default.payment.next.month',axis=1).copy()
    df3 = pd.concat([X, data_in], ignore_index = True)
    scaler = StandardScaler()
    df3 = pd.DataFrame(scaler.fit_transform(df3), columns=df3.columns)
    #print(data_in)
    data_in = df3.iloc[-1:]
    if loaded_model.predict(data_in) == 1:
        return "Customer will DEFAULT !!!"
    else:
        return "Customer will not default"
    