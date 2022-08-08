# -*- coding: utf-8 -*-
# @Date: 2022-07-28 20:16:06
# @Author: zhaorj
# @Version: 
# @Description: 

from test import TrainArgs, PredictSingleImgArgs, PredictImgsArgs, PredictArgs
from CCNet import train,predict_single_img,predict_imgs,predict

def add_function(a:int, b:int):
    return a + b


def main_train(args:TrainArgs):
    train.main(args)

def main_predict_single_img(args:PredictSingleImgArgs):
    predict_single_img.main(args)

def main_predict_imgs(args:PredictImgsArgs):
    predict_imgs.main(args)

def main_predict(args:PredictArgs):
    predict.main(args)



