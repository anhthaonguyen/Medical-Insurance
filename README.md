#importing Necessary Libraries

import numpy as np
import pandas as pd
import pickle as pkl 
import streamlit as st


model = pkl.load(open('MIPML.pkl', 'rb'))
scaler= pkl.load(open('my_scaler.pkl','rb'))

# Thiết lập giao diện Streamlit:
st.header('Medical Insurance Premium Predictor')

# Input fields for the user-Xử lý đầu vào:
gender = st.selectbox('Choose Gender', ['Female', 'Male'])
smoker = st.selectbox('Are you a smoker?', ['Yes', 'No'])
region = st.selectbox('Choose Region', ['SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'])
age = st.slider('Enter Age', 5, 80)
bmi = st.number_input('Enter BMI', min_value=5.0, max_value=100.0, value=25.0, step=0.1)
children = st.slider('Choose No of Children', 0, 5)

# Llogic xử lý khi người dùng nhấn nút Predict
if st.button('Predict'):
    # Xử lý biến đầu vàopip
    gender = 0 if gender == 'Female' else 1
    smoker = 1 if smoker == 'Yes' else 0
    # Mã hóa biến vùng miền
    if region == 'SouthEast':
        SouthEast, SouthWest, NorthEast, NorthWest = 1, 0, 0, 0
    elif region == 'SouthWest':
        SouthEast, SouthWest, NorthEast, NorthWest = 0, 1, 0, 0
    elif region == 'NorthEast':
        SouthEast, SouthWest, NorthEast, NorthWest = 0, 0, 1, 0
    else:  # NorthWest
        SouthEast, SouthWest, NorthEast, NorthWest = 0, 0, 0, 1
    # Chuẩn bị dữ liệu đầu vào cho mô hình:
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import boxcox
    lambda_bmi = 0.460304062303108  
    lambda_charges = 0.043516942579678274  # Thay bằng giá trị thực tế
    bmi_boxcox = boxcox(bmi, lambda_bmi)

    input_data = np.array([age, gender, bmi_boxcox, children, smoker, NorthEast, NorthWest, SouthEast, SouthWest])

    input_data_scaled = input_data.copy()
    
    input_data_scaled[[0, 2]] = scaler.transform(input_data[[0, 2]].reshape(-1, 2))[:, 0] #chỉ chuẩn hóa cột age và bmi
    
    input_data_scaled = input_data_scaled.reshape(1, -1)
    
    insurance_premium_boxcox = model.predict(input_data_scaled)
    
    insurance_premium = (insurance_premium_boxcox * lambda_charges + 1) ** (1 / lambda_charges)
    
    display_string = 'Insurance Premium will be '+ str(round(insurance_premium[0],2)) + ' USD Dollars'
    st.markdown(display_string)

