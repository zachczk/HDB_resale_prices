import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

# Add title and description
st.title('Predicting HDB Resale Property Prices')
st.markdown('Toy model to play to predict \
HDB resale prices')

# Add features sliders
st.header('HDB Property Features')
col1, col2 = st.columns(2)

with col1:
    st.text('Sepal characteristics')
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text('Pepal characteristics')
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

# Add prediction button
st.button('Predict HDB Property Price')
if st.button('Predict HDB Property Price'):
    result = predict(X_test_df)
    st.text(result)

st.text('')
st.text('')
st.markdown(
    '`Create by` [Zach Chen] [Linkedin: https://www.linkedin.com/in/zach-chen-73405167/ | \
         `Code:` [GitHub](https://github.com/zachczk)')
