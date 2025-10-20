import streamlit as st

page = st.navigation([
    st.Page('p1.py',title='IRIS相關資訊', icon='🌸'),
    st.Page('p2.py',title='單一分類器展示', icon='🌸'),
    st.Page('p3.py',title='iris品種預測', icon='🌸')
])

page.run()