import streamlit as st
import pandas as pd
import joblib
import random

model = joblib.load("logistic.pkl")
scaler = joblib.load("minmax.pkl")
hash_encoder = joblib.load("hashing_encoder.pkl")

state_options = [
    'OH', 'NJ', 'OK', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY',
    'ID', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'WY', 'HI', 'NH', 'AK', 'GA',
    'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'SD', 'NC', 'WA', 'MN',
    'NM', 'NV', 'DC', 'VT', 'KY', 'ME', 'MS', 'AL', 'NE', 'KS', 'TN', 'IL',
    'PA', 'CT', 'ND'
]

area_code_options = ['area_code_415', 'area_code_408', 'area_code_510']
international_plan_options = ['yes', 'no']

st.title("📊 Dự đoán khách hàng rời đi (Churn Prediction)")

# --- Thêm nút random dữ liệu ---
if 'random_clicked' not in st.session_state:
    st.session_state.random_clicked = False

if st.button("🎲 Random dữ liệu"):
    st.session_state.random_clicked = True

if st.session_state.random_clicked:
    # Random dữ liệu
    state = random.choice(state_options)
    area_code = random.choice(area_code_options)
    international_plan = random.choice(international_plan_options)
    account_length = random.randint(1, 3000)
    number_vmail_messages = random.randint(0, 50)
    total_day_minutes = round(random.uniform(0, 400), 1)
    total_day_calls = random.randint(0, 200)
    total_eve_minutes = round(random.uniform(0, 400), 1)
    total_eve_calls = random.randint(0, 200)
    total_night_minutes = round(random.uniform(0, 400), 1)
    total_night_calls = random.randint(0, 200)
    total_intl_minutes = round(random.uniform(0, 30), 1)
    total_intl_calls = random.randint(0, 20)
    number_customer_service_calls = random.randint(0, 10)
else:
    state = st.selectbox("Bang (State)", state_options)
    area_code = st.selectbox("Mã vùng (Area Code)", area_code_options)
    international_plan = st.selectbox("Có gói quốc tế không?", international_plan_options)
    account_length = st.number_input("Thời gian sử dụng tài khoản (ngày)", 0, 10000, 100)
    number_vmail_messages = st.number_input("Số tin nhắn hộp thư thoại", 0, 10000, 10)
    total_day_minutes = st.number_input("Tổng số phút gọi ban ngày", 0.0, 10000.0, 180.0, 0.1)
    total_day_calls = st.number_input("Tổng số cuộc gọi ban ngày", 0, 10000, 100)
    total_eve_minutes = st.number_input("Tổng số phút gọi buổi tối", 0.0, 10000.0, 200.0, 0.1)
    total_eve_calls = st.number_input("Tổng số cuộc gọi buổi tối", 0, 10000, 100)
    total_night_minutes = st.number_input("Tổng số phút gọi ban đêm", 0.0, 10000.0, 200.0, 0.1)
    total_night_calls = st.number_input("Tổng số cuộc gọi ban đêm", 0, 10000, 100)
    total_intl_minutes = st.number_input("Tổng số phút gọi quốc tế", 0.0, 10000.0, 10.0, 0.1)
    total_intl_calls = st.number_input("Tổng số cuộc gọi quốc tế", 0, 10000, 5)
    number_customer_service_calls = st.number_input("Số lần gọi tổng đài chăm sóc khách hàng", 0, 10000, 1)

# Chuyển đổi và chuẩn bị dữ liệu như bình thường
area_code_area_code_415 = 1 if area_code == 'area_code_415' else 0
area_code_area_code_510 = 1 if area_code == 'area_code_510' else 0
international_plan_yes = 1 if international_plan == 'yes' else 0

all_columns = [
    'state', 'area_code', 'international_plan', 'account_length', 'number_vmail_messages',
    'total_day_minutes', 'total_day_calls', 'total_eve_minutes', 'total_eve_calls',
    'total_night_minutes', 'total_night_calls', 'total_intl_minutes', 'total_intl_calls',
    'number_customer_service_calls', 'dummy_col_1', 'dummy_col_2', 'dummy_col_3',
    'dummy_col_4', 'dummy_col_5', 'dummy_col_6'
]

input_data = {col: 0 for col in all_columns}
input_data['state'] = state
input_data['account_length'] = account_length
input_data['number_vmail_messages'] = number_vmail_messages
input_data['total_day_minutes'] = total_day_minutes
input_data['total_day_calls'] = total_day_calls
input_data['total_eve_minutes'] = total_eve_minutes
input_data['total_eve_calls'] = total_eve_calls
input_data['total_night_minutes'] = total_night_minutes
input_data['total_night_calls'] = total_night_calls
input_data['total_intl_minutes'] = total_intl_minutes
input_data['total_intl_calls'] = total_intl_calls
input_data['number_customer_service_calls'] = number_customer_service_calls

df_input = pd.DataFrame([input_data])
state_hash = hash_encoder.transform(df_input)

final_df = pd.DataFrame({
    'col_0': state_hash.loc[0, 'col_0'],
    'col_1': state_hash.loc[0, 'col_1'],
    'col_2': state_hash.loc[0, 'col_2'],
    'col_3': state_hash.loc[0, 'col_3'],
    'col_4': state_hash.loc[0, 'col_4'],
    'col_5': state_hash.loc[0, 'col_5'],
    'col_6': state_hash.loc[0, 'col_6'],
    'col_7': state_hash.loc[0, 'col_7'],
    'account_length': account_length,
    'number_vmail_messages': number_vmail_messages,
    'total_day_minutes': total_day_minutes,
    'total_day_calls': total_day_calls,
    'total_eve_minutes': total_eve_minutes,
    'total_eve_calls': total_eve_calls,
    'total_night_minutes': total_night_minutes,
    'total_night_calls': total_night_calls,
    'total_intl_minutes': total_intl_minutes,
    'total_intl_calls': total_intl_calls,
    'number_customer_service_calls': number_customer_service_calls,
    'area_code_area_code_415': area_code_area_code_415,
    'area_code_area_code_510': area_code_area_code_510,
    'international_plan_yes': international_plan_yes
}, index=[0])

scale_columns = [
    'account_length', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
    'total_eve_minutes', 'total_eve_calls', 'total_night_minutes', 'total_night_calls',
    'total_intl_minutes', 'total_intl_calls', 'number_customer_service_calls'
]

final_df[scale_columns] = scaler.transform(final_df[scale_columns])

if st.button("🔍 Dự đoán"):
    pred = model.predict(final_df)[0]
    proba = model.predict_proba(final_df)[0][1]
    if pred == 1:
        st.error(f"⚠️ Khách hàng **CÓ THỂ RỜI BỎ** (Churn) với xác suất **{proba:.2%}**")
    else:
        st.success(f"✅ Khách hàng **KHÔNG RỜI BỎ** với xác suất **{1 - proba:.2%}**")
