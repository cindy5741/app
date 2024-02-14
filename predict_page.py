import streamlit as st
import pickle
import numpy as np 
import xgboost as xgb
import pandas as pd
from pandas import MultiIndex

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import plotly.figure_factory as ff
import plotly.express as px


import os

def load_model():
    filepath = os.path.abspath('Dementia_XGBModel_Packed_ver2.pkl')
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

# import joblib

# # 保存模型
# joblib.dump(model, 'model.pkl')

# 加载模型
# data= joblib.load('Dementia_XGBModel_Packed_ver2.pkl')


# def load_model():
#     with open('Dementia_XGBModel_Packed_ver2.pkl','rb') as file:
#         data=pickle.load(file)
#     return data

data=load_model()

model=data['model']


TH=data['Pred_TH']
feature_name=data['Feature_labels']


def show_predict_page():
    st.title("Prediction Result")
    st.sidebar.write("""### Input your data""")

    # HSEX={"Male","Female"}
    HSEX={0,1}
    Memory={0,1,2}
    Community_Affairsl={0,1,2,3}
    Personal_Care={0,1,2,3}
    MMSE_time={0,1,2,3,4,5}
    MMSE_place={0,1,2,3,4,5}
    MMSE_Recall={0,1,2,3}
    MMSE_Language={0,1,2,3,4,5,6,7,8,9}
    MMSE_Registration={0,1,2,3}
    MMSE_Attention_and_calculation={0,1,2,3,4,5}
    T4_normal={0,1,2}
    TSH_normal={0,1,2}
    BUN_normal={0,1}
    K_normal={0,1,2}
    CHOLESTEROL_normal={0,1}
    GPT_normal={0,1}
    eGFR_normal={0,1}
    # T4_normal={"Normal","High","Low"}
    # TSH_normal={"Normal","High","Low"}
    # BUN_normal={"Normal","High"}
    # K_normal={"Normal","High","Low"}
    # CHOLESTEROL_normal={"Normal","High"}
    # GPT_normal={"Normal","High"}
    # eGFR_normal={"Normal","Low"}
    hsex=st.sidebar.selectbox("HSEX (0=Male, 1=Female)",HSEX)
    memory=st.sidebar.selectbox("Memory (CDR)",Memory)
    community_affairsl=st.sidebar.selectbox("Community_Affairsl (CDR)",Community_Affairsl)
    personal_care=st.sidebar.selectbox("Personal_Care (CDR)",Personal_Care)
    mmse_time=st.sidebar.selectbox("MMSE_time",MMSE_time)
    mmse_place=st.sidebar.selectbox("MMSE_place",MMSE_place)
    mmse_recall=st.sidebar.selectbox("MMSE_Recall", MMSE_Recall)
    mmse_language=st.sidebar.selectbox("MMSE_Language",MMSE_Language)
    mmse_registration=st.sidebar.selectbox("MMSE_Registration",MMSE_Registration)
    mmse_attention_and_calculation=st.sidebar.selectbox("MMSE_Attention_and_calculation",MMSE_Attention_and_calculation)
    t4_normal=st.sidebar.selectbox("T4_normal",T4_normal)
    tsh_normal=st.sidebar.selectbox("TSH_normal",TSH_normal)
    bun_normal=st.sidebar.selectbox("BUN_normal",BUN_normal)
    k_normal=st.sidebar.selectbox("K_normal",K_normal)
    cholesterol_normal=st.sidebar.selectbox("CHOLESTEROL_normal",CHOLESTEROL_normal)
    gpt_normal=st.sidebar.selectbox("GPT_normal",GPT_normal)
    egfr_normal=st.sidebar.selectbox("eGFR_normal",eGFR_normal)

    #code for predict
    dementia=" "

    ok=st.sidebar.button("Predict")
    if ok:
        X=np.array([[mmse_time,mmse_place,egfr_normal,mmse_recall,t4_normal,
                     mmse_attention_and_calculation,personal_care,memory,mmse_language,tsh_normal,
                     hsex,mmse_registration,bun_normal,k_normal,community_affairsl,
                     cholesterol_normal,gpt_normal
                     ]])
       
        X=pd.DataFrame(X, columns=feature_name)
        st.write("Prediction Outcome: ")
        dementia=model.predict_proba(X)[:,1].round(4)
        # dementia0=model.predict_proba(X)[:,0].round(4)
        # dementia1=model.predict_proba(X)[:,1].round(4)
        # 四位
        if (dementia<TH):
            
            st.success('Negative') 
        else:
            
            st.success('Positive') 
            
        st.write("Prediction Likelihood:")
        st.success(dementia[0])
        # st.success(dementia0)
        # st.success(dementia1)
        

        n_df=pd.read_csv("Negative_Likelihood.csv")
        p_df=pd.read_csv("Positive_Likelihood.csv")

        # print(df['negative'])
        n_data=n_df['positive']
        p_data=p_df['positive']

        # 計算 KDE
        kde = gaussian_kde(n_data)
        p_kde = gaussian_kde(p_data)
        # 設定 KDE 計算的 x 範圍
        n_x = np.linspace(min(n_data) - 0.1, max(n_data) + 0.1, 1000)
        p_x = np.linspace(min(p_data) - 0.1, max(p_data) + 0.1, 1000)

        # 計算對應的密度值
        density = kde(n_x)

        # 計算對應的密度值
        p_density = p_kde(p_x)

        # 創建 Matplotlib 圖形物件
        fig, ax = plt.subplots()



        # 繪製 negative KDE 圖
        ax.plot(n_x, density)  # KDE 圖
        ax.fill_between(n_x, 0, density, color='skyblue', alpha=0.5, label='Negative Likelihood distribution')  # 填充顏色區域
        
        # 繪製 positive KDE 圖
        ax.plot(p_x, p_density)  # KDE 圖
        ax.fill_between(p_x, 0, p_density, color='pink', alpha=0.5, label='Positive Likelihood distribution')  # 填充顏色區域

        # 資料散佈點在 KDE 圖上的位置
        # densityData = kde(dementia)

        p_densityData = p_kde(dementia)
        
        
        
        
        # ax.scatter(dementia,densityData,label='Prediction',color='red', alpha=1)  # 資料散佈點
        ax.scatter(dementia,p_densityData,color='red', alpha=1)  # 資料散佈點
        ax.axvline(x=TH, color='black', linestyle='--', label='Threshold: 0.32')  # 紅色虛線
        # ax.axvline(x=dementia, color='green', linestyle='--', label='pred_TH')  # 紅色虛線
        # ax.axvline(x=0.3209773, color='red', linestyle='--', label='TH')  # 紅色虛線

        # 加入標籤和標題
        ax.set_xlabel('Likelihood')
        ax.set_ylabel('Density')
        ax.set_title('Probability distribution of the trained model')

        # 顯示圖例
        ax.legend()

        # 顯示 KDE 圖
        st.pyplot(fig)
        
        # 设置 Seaborn 样式
        sns.set(style='darkgrid')

        # 创建 Matplotlib 图形对象
        fig, ax = plt.subplots(figsize=(20, 12), dpi=600)

        
        # 绘制正类别的直方图和 KDE 曲线
        sns.histplot(x=p_data, color='red', kde=True, stat='probability', label='Positive likelihood')
        # sns.lineplot(x=p_x, y=p_density, color='blue', label='KDE for Positive')

        # 绘制负类别的直方图和 KDE 曲线
        sns.histplot(x=n_data, color='blue', kde=True, stat='probability', label='Negative likelihood')
        # sns.lineplot(x=n_x, y=density, color='green', label='KDE for Negative')

        # 轴标签
        plt.xlabel('Predicted Likelihood', fontsize=18)
        plt.ylabel('Probability', fontsize=18)
        plt.legend(fontsize=16)

        # 显示图形
        # st.pyplot(fig)



show_predict_page()



