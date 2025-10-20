import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn import datasets

st.title('IRIS品種預測')
knn = joblib.load('knn.joblib')
svm = joblib.load('svm.joblib')
RF = joblib.load('RF.joblib')
LR = joblib.load('LR.joblib')

m = st.sidebar.selectbox('請選擇分類模型',["KNN",'SVM','Random Forest', 'Logistic Regression'])

if m == 'KNN':
    model = knn
elif m == 'SVM':
    model = svm
elif m == 'Random Forest':
    model = RF
else:
    model = LR

#接受預測資料
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
se1 = st.slider("#### 花萼長度", 
                float(df['sepal length (cm)'].min()),
                float(df['sepal length (cm)'].max()),
                float(df['sepal length (cm)'].mean()))

se2 = st.slider("#### 花萼寬度", 
                float(df['sepal width (cm)'].min()),
                float(df['sepal width (cm)'].max()),
                float(df['sepal width (cm)'].mean()))

se3 = st.slider("#### 花瓣長度", 
                float(df['petal length (cm)'].min()),
                float(df['petal length (cm)'].max()),
                float(df['petal length (cm)'].mean()))

se4 = st.slider("#### 花瓣寬度", 
                float(df['petal width (cm)'].min()),
                float(df['petal width (cm)'].max()),
                float(df['petal width (cm)'].mean()))

#預測
labels = ['Setosa','Versicolor','Virginica']
if st.button("預測"):
    x = [[se1, se2, se3, se4]]
    y_pred = model.predict(x)
    st.success(f'預測的品種為:{labels[y_pred[0]]}')