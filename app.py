import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

# Add title and description
st.title('Predicting HDB Resale Property Prices')
st.markdown('Toy model to play to predict HDB resale prices')

# Add features sliders
st.header('HDB Property Features')
col1, col2 = st.columns(2)

with col1:
    remaining_lease_months = st.slider('Remaining Lease (Months)', 500, 1200, 10)
    floor_area_sqm = st.slider('Floor Area (sqm)', 30, 250, 10)

# Add dropdown boxes
with col2:
    town = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
            'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
            'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
            'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
            'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
            'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    selected_town = st.selectbox('Selected Town', town)
    st.write('You selected:', selected_town)

    storey_range = ['10 TO 12', '01 TO 03', '04 TO 06', '07 TO 09', '13 TO 15',
                    '19 TO 21', '22 TO 24', '16 TO 18', '34 TO 36', '28 TO 30',
                    '37 TO 39', '49 TO 51', '25 TO 27', '40 TO 42', '31 TO 33',
                    '46 TO 48', '43 TO 45']
    selected_storey_range = st.selectbox('Selected Storey Range', storey_range)
    st.write('You selected:', selected_storey_range)


# # Preprocess data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
remaining_lease_months_scaled = scaler.fit_transform([remaining_lease_months])
floor_area_sqm_scaled = scaler.fit_transform([floor_area_sqm])

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
selected_town =  encoder.fit_transform([selected_town])
selected_storey_range =  encoder.fit_transform([selected_storey_range])


# Add prediction button
if st.button('Predict HDB Property Price', key='prediction_button'):
    result = predict(remaining_lease_months, floor_area_sqm, selected_town, selected_storey_range)
    st.text(result)

    st.text('')
    st.text('')
    st.markdown(
    '`Create by` [Zach Chen] [Linkedin: https://www.linkedin.com/in/zach-chen-73405167/ | \
         `Code:` [GitHub](https://github.com/zachczk)')