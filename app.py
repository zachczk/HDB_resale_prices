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


# # Preprocess data input
town_mapping = {
    'ANG MO KIO': 0,
    'BEDOK': 1,
    'BISHAN': 2,
    'BUKIT BATOK': 3,
    'BUKIT MERAH': 4,
    'BUKIT PANJANG': 5,
    'BUKIT TIMAH': 6,
    'CENTRAL AREA': 7,
    'CHOA CHU KANG': 8,
    'CLEMENTI': 9,
    'GEYLANG': 10,
    'HOUGANG': 11,
    'JURONG EAST': 12,
    'JURONG WEST': 13,
    'KALLANG/WHAMPOA': 14,
    'MARINE PARADE': 15,
    'PASIR RIS': 16,
    'PUNGGOL': 17,
    'QUEENSTOWN': 18,
    'SEMBAWANG': 19,
    'SENGKANG': 20,
    'SERANGOON': 21,
    'TAMPINES': 22,
    'TOA PAYOH': 23,
    'WOODLANDS': 24,
    'YISHUN': 25
}

if selected_town in town_mapping:
    selected_town = town_mapping[selected_town]
    

storey_mapping = {
    '01 TO 03': 0,
    '04 TO 06': 1,
    '07 TO 09': 2,
    '10 TO 12': 3,
    '13 TO 15': 4,
    '16 TO 18': 5,
    '19 TO 21': 6,
    '22 TO 24': 7,
    '25 TO 27': 8,
    '28 TO 30': 9,
    '31 TO 33': 10,
    '34 TO 36': 11,
    '37 TO 39': 12,
    '40 TO 42': 13,
    '43 TO 45': 14,
    '46 TO 48': 15,
    '49 TO 51': 16
}

if selected_storey_range in storey_mapping:
    selected_storey_range = storey_mapping[selected_storey_range]


from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the saved scaler
scaler = joblib.load('scaler.joblib')

categorical_features = []  # Example categorical feature names
numerical_features = []

data = {
    'remaining_lease_months': [remaining_lease_months],
    'floor_area_sqm': [floor_area_sqm],
    'town': [selected_town],
    'storey_range': [selected_storey_range]
}

X_df = pd.DataFrame(data, index=[0]) 

X_scaled_array = scaler.transform(X_df)

X_scaled_df = pd.DataFrame(X_scaled_array, columns = X_df.columns)


# Add prediction button
if st.button('Predict HDB Property Price', key='prediction_button'):
    result = predict(remaining_lease_months, floor_area_sqm, selected_town, selected_storey_range)
    st.text(result)

    st.text('')
    st.text('')
    st.markdown(
    '`Create by` [Zach Chen] [Linkedin: https://www.linkedin.com/in/zach-chen-73405167/ | \
         `Code:` [GitHub](https://github.com/zachczk)')