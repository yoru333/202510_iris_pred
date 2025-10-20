import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.title('IRIS資料集展示')
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
st.write(df.head())

colors = ['r', 'g', 'b']
df['target'] = iris.target
mapping = {'setosa':0, 'versicolor':1, 'virginica':2}
tab1, tab2 = st.tabs(['依花萼的長寬','依花瓣的長寬'])
fig, ax = plt.subplots()

with tab1:
    for i, s in mapping.items():
        subset = df[df['target'] == s]
        ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                   label=i, c=colors[s])
    ax.set_xlabel('sepal length (cm)') 
    ax.set_ylabel('sepal width (cm)')
    ax.legend()
    st.pyplot(fig)   


fig2, ax2 = plt.subplots()
with tab2:
    for i, s in mapping.items():
        subset = df[df['target'] == s]
        ax2.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                   label=i, c=colors[s])
    ax2.set_xlabel('petal length (cm)') 
    ax2.set_ylabel('petal width (cm)')
    ax2.legend()
    st.pyplot(fig2)   

st.write('### PCA:將全部特徵轉換成二維')
X = iris.data
scaler = StandardScaler()
X_scaler = pd.DataFrame(scaler.fit_transform(X), columns=iris.feature_names)

pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(X_scaler)

fig3, ax3 = plt.subplots()

y = df['target']
for i, name in enumerate(y.unique()):
    m = (y.values==i)
    ax3.scatter(x_pca[m, 0], x_pca[m, 1], label=name, c=colors[i])
ax3.set_xlabel('PCA1')
ax3.set_ylabel('PCA2')
ax3.set_title('PCA OF IRIS Dataset')
ax3.legend()
st.pyplot(fig3)





