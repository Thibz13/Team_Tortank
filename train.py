import torch
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as dec

from sklearn.model_selection import KFold

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from  xgboost import XGBRegressor

from sklearn.model_selection import train_test_split


print("data importation")
coef_cl_raw = pd.read_csv(r'coef_city_lang.csv',sep = ",")
coef_cl =dec(coef_cl_raw)

coef_list_base_raw = pd.read_csv(r'list_base_price.csv',sep = ",")
coef_list_base =dec(coef_list_base_raw)

df_raw = pd.read_csv(r'data_all_extended_pourc_stock.csv',sep = ";")
df =dec(df_raw)


coef_cl_dico = {(city,lang): float(coef_cl[ (coef_cl.city == city) & (coef_cl.lang == lang) ]["coef"]) for city in coef_cl.city for lang in coef_cl.lang }

coef_list_base_dico = {(hotel_id): float(coef_list_base[ (coef_list_base.hotel_id == hotel_id) ]["base_price"]) for hotel_id in coef_list_base.hotel_id}

coef_list_stock_dico = {(hotel_id): int(coef_list_base[ (coef_list_base.hotel_id == hotel_id) ]["stock"]) for hotel_id in coef_list_base.hotel_id}

coef_order_request_city_list = [1.0,1.001,1.034,1.048,1.049]

print("data featuring")
df_norm = dec(df_raw)
df_norm = df_norm[df_norm["date"] <= 44]

df_norm["pour_price"] = df_norm.apply(lambda row: row.price/coef_cl_dico[row.city,row.language] /
                            coef_order_request_city_list[min(4,row.order_request_city)]/ coef_list_base_dico[row.hotel_id], axis = 1)

jeu_norm = dec(df_norm[["pour_price","pourc_stock","hotel_id","date"]]).reset_index()
jeu_norm.pop("index")



L_mod = [XGBRegressor(n_estimators=2500, max_depth=9, eta=0.1, subsample=0.7, colsample_bytree=0.8),
        XGBRegressor(n_estimators=2500, max_depth=9, eta=0.1, subsample=0.7, colsample_bytree=0.8),
        XGBRegressor(n_estimators=2500, max_depth=9, eta=0.1, subsample=0.7, colsample_bytree=0.8)]

kf = KFold(n_splits=len(L_mod), random_state=None, shuffle=True)

print(" we are now going to train the 3 models")

for i, (train_index, _) in enumerate(kf.split(jeu_norm),start=0):
    
    print("training model "+ str(i) +" :",end="")
    my_jeu_norm = dec(jeu_norm.loc[train_index]    )
    
    Y = my_jeu_norm.pop('pour_price')
    X = my_jeu_norm    
    L_mod[i].fit(X.values, Y.values)
    print(" Done")

print("model saving")
for i in range(len(L_mod)):    L_mod[i].save_model("model_sklearn_"+str(i)+".json")
print("the saving is done, you can now launch the gradio app")

    