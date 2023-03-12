import xgboost as xgb
import gradio as gr
import pandas as pd 
import numpy as np
from copy import deepcopy as dec


test_set_raw = pd.read_csv(r'test_set.csv',sep = ";")

coef_cl_raw = pd.read_csv(r'coef_city_lang.csv',sep = ",")
coef_cl =dec(coef_cl_raw)

coef_cl_dico = {(city,lang): float(coef_cl[ (coef_cl.city == city) & (coef_cl.lang == lang) ]["coef"]) for city in coef_cl.city for lang in coef_cl.lang }



coef_list_base_raw = pd.read_csv(r'list_base_price.csv',sep = ",")
coef_list_base =dec(coef_list_base_raw)

coef_list_base_dico = {(hotel_id): float(coef_list_base[ (coef_list_base.hotel_id == hotel_id) ]["base_price"]) for hotel_id in coef_list_base.hotel_id}

coef_list_stock_dico = {(hotel_id): int(coef_list_base[ (coef_list_base.hotel_id == hotel_id) ]["stock"]) for hotel_id in coef_list_base.hotel_id}

coef_order_request_city_list = [1.0,1.001,1.034,1.048,1.049]

def bridge(data_table):
    
    data_table["pourc_stock"] = data_table.apply(lambda row: 
            row.stock/coef_list_stock_dico[row.hotel_id], axis = 1)
    
    test_set_norm = dec(data_table[["pourc_stock","hotel_id","date"]])

    L_mod = []
    pred = np.zeros(len(data_table))
    
    for j in range(3):
        L_mod.append( xgb.XGBRegressor()) 
        L_mod[j].load_model("model_sklearn_"+str(j)+".json")
        pred += L_mod[j].predict(test_set_norm)
            
    data_table["pred_1"] = pred / 3
    
    list_request = {idx : [] for idx in np.unique(data_table.avatar_id)}
    
    for index, row in data_table.iterrows():
        list_request[row['avatar_id']].append(row["order_requests"])
    
    for j in list_request : 
        list_request[j] = np.unique(list_request[j])
        list_request[j] = list(list_request[j])
        
    data_table["final_pred"] = data_table.apply(
               lambda row: row.pred_1 * coef_cl_dico[row.city,row.language] *
                coef_order_request_city_list[min([list_request[row.avatar_id].index(row.order_requests),4])] * 
                coef_list_base_dico[row.hotel_id] , axis = 1)


    
    
    submit_me =  pd.DataFrame( [round(x) for x in data_table["final_pred"]], columns = ["price"] )
    
    return submit_me

inputs = [gr.Dataframe(label="Input Data", interactive=0)]

outputs = [gr.Dataframe(label="Predictions")]

gr.Interface(fn = bridge, inputs = inputs, outputs = outputs,examples = [[test_set_raw.head(10)],[test_set_raw.head(100)],[test_set_raw.head(1000)],[test_set_raw]]).launch(share = True)