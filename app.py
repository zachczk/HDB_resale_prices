import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict #predict is a function inthe prediction script


# Add title and description
st.title('Predicting HDB Resale Property Prices')
st.markdown('Toy model to play to predict HDB resale prices')

# Add features sliders
st.header('HDB Property Features')
col1, col2 = st.columns(2)

with col1:
    remaining_lease_input = st.text_input('Remaining Lease (Months)')
    floor_area_input = st.text_input('Floor Area (sqm)')

    try:
        remaining_lease_months = int(remaining_lease_input)
        floor_area_sqm = int(floor_area_input)
    except ValueError:
        st.error('Please enter valid numeric values.')
        st.stop()

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

    storey_range = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15',
                    '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30',
                    '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45',
                    '46 TO 48', '49 TO 51']
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

def scale_features(df):
    scaled_df = pd.DataFrame()
    for column in df.columns:
        max_value = df[column].max()
        min_value = df[column].min()
        
        # Check if the range is zero
        if max_value == min_value:
            scaled_df[column] = df[column]  # Set the column as it is
        else:
            scaled_df[column] = (df[column] - min_value) / (max_value - min_value)
    return scaled_df

data = {
    'remaining_lease_months': [remaining_lease_months],
    'floor_area_sqm': [floor_area_sqm],
    'town': [selected_town],
    'storey_range': [selected_storey_range]
}

X_df = pd.DataFrame(data, index=[0]) 

X_scaled_df = scale_features(X_df)


# Add prediction button
if st.button('Predict HDB Property Price', key='prediction_button'):
    result = predict(X_scaled_df)
    scalar_result = result.item()
    formatted_result = '${:,.0f}'.format(scalar_result)
    st.write(formatted_result)

    st.text('')
    st.text('')
    st.markdown(
    '`Create by` [Zach Chen] [Linkedin: https://www.linkedin.com/in/zach-chen-73405167/ | \
         `Code:` [GitHub](https://github.com/zachczk)')