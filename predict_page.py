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
from collections import OrderedDict

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
    st.sidebar.markdown('<b style="font-size:28px; ">Machine Learning Prediction Model for Dementia Progression with Multiple Integrated Data</b>', unsafe_allow_html=True)
    st.sidebar.write("""### Input your data""")

    # HSEX={"Male","Female"}
    HSEX={0,1}
    Memory=OrderedDict([(0, 0), (0.5, 0.5), (1, 1), (2, 2), (3, 3)])
    Community_Affairsl=OrderedDict([(0, 0), (0.5, 0.5), (1, 1), (2, 2), (3, 3)])
    Personal_Care=OrderedDict([(0, 0), (0.5, 0.5), (1, 1), (2, 2), (3, 3)])
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
    # T4_normal={"Normal":0,"High":1,"Low":2}
    # TSH_normal={"Normal":0,"High":1,"Low":2}
    # BUN_normal={"Normal","High"}
    # K_normal={"Normal":0,"High":1,"Low":2}
    # CHOLESTEROL_normal={"Normal":0,"High":1}
    # GPT_normal={"Normal":0,"High":1}
    # eGFR_normal={"Normal":0,"Low":1}
    hsex=st.sidebar.selectbox("Sex (0=Male, 1=Female)",HSEX)
    memory=st.sidebar.selectbox("Memory (CDR)",Memory)
    community_affairsl=st.sidebar.selectbox("Community affairs (CDR)",Community_Affairsl)
    personal_care=st.sidebar.selectbox("Personal care (CDR)",Personal_Care)
    mmse_time=st.sidebar.selectbox("MMSE time",MMSE_time)
    mmse_place=st.sidebar.selectbox("MMSE place",MMSE_place)
    mmse_recall=st.sidebar.selectbox("MMSE recall", MMSE_Recall)
    mmse_language=st.sidebar.selectbox("MMSE language",MMSE_Language)
    mmse_registration=st.sidebar.selectbox("MMSE registration",MMSE_Registration)
    mmse_attention_and_calculation=st.sidebar.selectbox("MMSE attention and calculation",MMSE_Attention_and_calculation)
    t4_normal=st.sidebar.selectbox("T4 (0=Normal, 1=High, 2=Low)",T4_normal)
    tsh_normal=st.sidebar.selectbox("TSH (0=Normal, 1=High, 2=Low)",TSH_normal)
    bun_normal=st.sidebar.selectbox("BUN (0=Normal, 1=High)",BUN_normal)
    k_normal=st.sidebar.selectbox("K (0=Normal, 1=High, 2=Low)",K_normal)
    cholesterol_normal=st.sidebar.selectbox("Cholesterol (0=Normal, 1=High)",CHOLESTEROL_normal)
    gpt_normal=st.sidebar.selectbox("GPT (0=Normal, 1=High)",GPT_normal)
    egfr_normal=st.sidebar.selectbox("eGFR (0=Normal, 1=Low)",eGFR_normal)

    #code for predict
    dementia=" "

    ok=st.sidebar.button("Predict")
    st.sidebar.write("Category variables in our prediction training model were classified according to the reference values in Fu Jen Catholic University Hospital, as demonstrated in Supplement Table 1. Users can enter category variables data based on their medical units' recommendations and clinical judgment.")
    # st.sidebar.markdown("<p style='font-size: 12px;'>Category variables in our prediction training model were classified according to the reference values in Fu Jen Catholic University Hospital, as demonstrated in Supplement Table 1. Users can enter category variables data based on their medical units' recommendations and clinical judgment.</p>", unsafe_allow_html=True)
    if ok:
        X=np.array([[mmse_time,mmse_place,egfr_normal,mmse_recall,t4_normal,
                     mmse_attention_and_calculation,personal_care,memory,mmse_language,tsh_normal,
                     hsex,mmse_registration,bun_normal,k_normal,community_affairsl,
                     cholesterol_normal,gpt_normal
                     ]])
       
        X=pd.DataFrame(X, columns=feature_name)
        
        dementia=model.predict_proba(X)[:,1].round(4)
        # dementia0=model.predict_proba(X)[:,0].round(4)
        # dementia1=model.predict_proba(X)[:,1].round(4)
        # 四位
        st.write("Prediction Likelihood: (Threshold: 0.32)")
        st.success(dementia[0])

        # st.write("Threshold: 0.32")

        st.write("Predict Progression of Dementia: ")
        if (dementia<TH):
            
            st.success('Negative') 
        else:
            
            st.success('Positive') 
            
        
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
        ax.axvline(x=TH, color='black', linestyle='--', label='Threshold: 0.32')  # 黑色虛線
        ax.axvline(x=dementia, color='red', linestyle='--')  # 紅色虛線
        # ax.axvline(x=0.3209773, color='red', linestyle='--', label='TH')  # 紅色虛線
        
        ax.set_xlim(0, 1)
        # 设置 y 轴范围从 0 开始往上
        ax.set_ylim(0, None)
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        
        # 加入標籤和標題
        ax.set_xlabel('Likelihood')
        ax.set_ylabel('Density')
        # ax.set_title('Probability distribution of the trained model')

        # 顯示圖例
        ax.legend()

        # 顯示 KDE 圖
        st.pyplot(fig)
        


show_predict_page()



