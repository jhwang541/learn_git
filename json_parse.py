#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:40:23 2018

@author: mitarashi
"""

import pandas as pd
import numpy as np
import json
from time import time

data1 = pd.read_excel(r'/Users/mitarashi/Documents/your-data.xlsx')

### 对str格式的json列进行解析，并存储到新的列中，删除原始列
list1 = [json.loads(i)['data'] for i in data1['detail']]
data1['data'] = list1
data1.drop('detail',axis=1,inplace=True)

### 对嵌套的json进行拆包，每次拆一层
def json_to_columns(df,col_name):
    for i in df[col_name][0].keys():         # 对dict的第一层key进行循环
        list2=[j[i] for j in df[col_name]]   # 存储对应上述key的value至列表推导式
        df[i]=list2                          # 存储到新的列中
    df.drop(col_name,axis=1,inplace=True)    # 删除原始列
    return df

### 遍历整个dataframe，处理所有值类型为dict的列
def json_parse(df):
    for i in df.keys():
        if type(df[i][0])==dict:
            df=json_to_columns(df,i)            
    return df

### 特殊情况。处理item类型为list的列，转换为dict
def list_parse(df):
    for i in df.keys():
        if type(df[i][0])==list:
            list1=[j[0] if j!=[] else {} for j in df[i]]
            df[i]=list1
    return df

#重复调用，直至所有dict都被拆解
data1=json_parse(data1)
data1=list_parse(data1)