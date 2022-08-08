# -*- coding: utf-8 -*-
# @Date: 2022-06-10 15:08:00
# @Author: zhaorj
# @Version: 
# @Description: 

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
import uvicorn# type: ignore
from pydantic import BaseModel
from typing import List, Optional
import requests
from apps import (
    main_predict_single_img,
    main_predict_imgs,
    main_predict,
    main_train,
    add_function,
)

permission_url = 'http://localhost:8000/permission'
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def permission_required(token: str = Depends(oauth2_scheme), model_name: str = 'predict'):
    r = requests.post(permission_url, headers={'Authorization': f'Bearer {token}'}, json={'model_name': model_name})
    if r.status_code == 200:
        return True, r.json()
    return False

app = FastAPI()

class User(BaseModel):
    username: str
    role: str = 'user'
    disabled: Optional[bool] = None

class Add_Item(BaseModel):
    a: int
    b: int

class Model(BaseModel):
    model_name:str


from pydantic import BaseModel
from test import TrainArgs, PredictSingleImgArgs, PredictImgsArgs, PredictArgs


# # 向权限管理系统发送验证消息，判断用户是否有权限访问
# def send_request(url:str, data:dict):
#     headers = {'Content-Type': 'application/json'}
#     response = requests.post(url, json=data, headers=headers)
#     return response.json()


# def authorization(token, model):
#     url = "http://localhost:8000/api/v1/authorization"
#     data = {"token": token, "model": model}
#     res = requests.post(url, json=data)
#     return res.json()


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/add/{a}/{b}")
def add(a: int, b: int):
    return {"result": a + b}


@app.post("/add/")
def add_post(item: Add_Item):
    return {"result": add_function(item.a, item.b)}

@app.post("/file/")
def file_upload(file: UploadFile):
    return {"file": file.filename}


@app.get("/html/", response_class=HTMLResponse)
async def server():
    html_file = open(r"html\index.html", 'r').read()
    return html_file


@app.post("/uploadfile/")
async def uploadfile(file: List[UploadFile], token: str = Depends(oauth2_scheme)):
    if not permission_required(token):
        return HTTPException(
            status_code = 401,
            detail = "You don't have permission to access this resource"
        )

    for f in file:
        if f.filename.endswith(".jpg"):
            test = await f.read()
            with open('test.jpg', 'wb') as f:
                f.write(test)
        if f.filename.endswith(".tiff"):
            test = await f.read()
            with open('test.tiff', 'wb') as f:
                f.write(test)
    return {"file": file[0].filename, "type": file[0].content_type}


# @app.post("/predict/")
# async def predict(args:PredictArgs):


# @app.post("/train/")
# def train(args:TrainArgs):
#     main_train(args)
#     return {"message": "train success"}

# @app.post("/predict_single_img/")
# def predict_single_img(args:PredictSingleImgArgs):
#     main_predict_single_img(args)
#     return {"message": "predict_single_img success"}

# @app.post("/predict_imgs/")
# def predict_imgs(args:PredictImgsArgs):
#     main_predict_imgs(args)
#     return {"message": "predict_imgs success"}

# @app.post("/predict/")
# def predict(args:PredictArgs):
#     main_predict(args)
#     return {"message": "predict success"}


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8080, debug=True)






