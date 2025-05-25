import streamlit as st
import pandas as pd
import joblib
import category_encoders as ce
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load model và scaler
model = joblib.load("logistic.pkl")
scaler = joblib.load("minmax.pkl")

# Danh sách state gốc
state_options = [
    'OH', 'NJ', 'OK', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY',
    'ID', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'WY', 'HI', 'NH', 'AK', 'GA',
    'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'SD', 'NC', 'WA', 'MN',
    'NM', 'NV', 'DC', 'VT', 'KY', 'ME', 'MS', 'AL', 'NE', 'KS', 'TN', 'IL',
    'PA', 'CT', 'ND'
]

area_code_options = ['area_code_415', 'area_code_408', 'area_code_510']
international_plan_options = ['yes', 'no']

# Khởi tạo HashingEncoder với 8 thành phần
hash_encoder = ce.HashingEncoder(cols=['state'], n_components=8)
hash_encoder.fit(pd.DataFrame({'state': state_options}))

st.set_page_config(page_title="Dự đoán Churn", layout="centered")
st.title("📊 Dự đoán khách hàng rời đi (Churn Prediction)")

st.subheader("1️⃣ Thông tin khách hàng")

# Hàm tạo dữ liệu ngẫu nhiên
def random_data():
    data = {}
    data['state'] = np.random.choice(state_options)
    data['area_code'] = np.random.choice(area_code_options)
    data['international_plan'] = np.random.choice(international_plan_options)

    # Các giá trị số liệu trong phạm vi hợp lý
    data['account_length'] = np.random.randint(1, 300)
    data['number_vmail_messages'] = np.random.randint(0, 50)
    data['total_day_minutes'] = round(np.random.uniform(0, 400), 1)
    data['total_day_calls'] = np.random.randint(0, 200)
    data['total_eve_minutes'] = round(np.random.uniform(0, 400), 1)
    data['total_eve_calls'] = np.random.randint(0, 200)
    data['total_night_minutes'] = round(np.random.uniform(0, 400), 1)
    data['total_night_calls'] = np.random.randint(0, 200)
    data['total_intl_minutes'] = round(np.random.uniform(0, 20), 1)
    data['total_intl_calls'] = np.random.randint(0, 20)
    data['number_customer_service_calls'] = np.random.randint(0, 10)
    return data

# Nút random dữ liệu
if st.button("🎲 Random dữ liệu"):
    rand_data = random_data()
    st.session_state['state'] = rand_data['state']
    st.session_state['area_code'] = rand_data['area_code']
    st.session_state['international_plan'] = rand_data['international_plan']
    for key in rand_data:
        if key not in ['state', 'area_code', 'international_plan']:
            st.session_state[key] = rand_data[key]

# Sử dụng session_state để giữ giá trị khi random
state = st.selectbox("Bang (State)", state_options, index=state_options.index(st.session_state.get('state', state_options[0])))

area_code = st.selectbox("Mã vùng (Area Code)", area_code_options, index=area_code_options.index(st.session_state.get('area_code', area_code_options[0])))
area_code_area_code_415 = 1 if area_code == 'area_code_415' else 0
area_code_area_code_510 = 1 if area_code == 'area_code_510' else 0

international_plan = st.selectbox("Có gói quốc tế không?", international_plan_options,
                                  index=international_plan_options.index(st.session_state.get('international_plan', 'no')))
international_plan_yes = 1 if international_plan == 'yes' else 0

st.subheader("2️⃣ Dữ liệu dịch vụ (Nhập số)")

feature_defaults = {
    'account_length': 100,
    'number_vmail_messages': 10,
    'total_day_minutes': 180.0,
    'total_day_calls': 100,
    'total_eve_minutes': 200.0,
    'total_eve_calls': 100,
    'total_night_minutes': 200.0,
    'total_night_calls': 100,
    'total_intl_minutes': 10.0,
    'total_intl_calls': 5,
    'number_customer_service_calls': 1
}

# Tên mô tả dễ hiểu cho các feature
feature_labels = {
    'account_length': "Thời gian sử dụng tài khoản (ngày)",
    'number_vmail_messages': "Số tin nhắn hộp thư thoại",
    'total_day_minutes': "Tổng số phút gọi ban ngày",
    'total_day_calls': "Tổng số cuộc gọi ban ngày",
    'total_eve_minutes': "Tổng số phút gọi buổi tối",
    'total_eve_calls': "Tổng số cuộc gọi buổi tối",
    'total_night_minutes': "Tổng số phút gọi ban đêm",
    'total_night_calls': "Tổng số cuộc gọi ban đêm",
    'total_intl_minutes': "Tổng số phút gọi quốc tế",
    'total_intl_calls': "Tổng số cuộc gọi quốc tế",
    'number_customer_service_calls': "Số lần gọi tổng đài chăm sóc khách hàng"
}

feature_inputs = {}
for feature, default_val in feature_defaults.items():
    label = feature_labels.get(feature, feature)
    value = st.session_state.get(feature, default_val)
    step_val = 0.1 if "minutes" in feature else 1
    if isinstance(default_val, float):
        val = st.number_input(label, min_value=0.0, max_value=10000.0, value=value, step=step_val, format="%.1f")
    else:
        val = st.number_input(label, min_value=0, max_value=10000, value=value, step=step_val)
    feature_inputs[feature] = val

if st.button("🔍 Dự đoán"):

    # Encode state (HashingEncoder trả về 8 cột col_0..col_7)
    state_df = pd.DataFrame({'state': [state]})
    state_hash = hash_encoder.transform(state_df)

    input_df = pd.DataFrame([{
        'col_0': state_hash.loc[0, 'col_0'],
        'col_1': state_hash.loc[0, 'col_1'],
        'col_2': state_hash.loc[0, 'col_2'],
        'col_3': state_hash.loc[0, 'col_3'],
        'col_4': state_hash.loc[0, 'col_4'],
        'col_5': state_hash.loc[0, 'col_5'],
        'col_6': state_hash.loc[0, 'col_6'],
        'col_7': state_hash.loc[0, 'col_7'],
        **feature_inputs,
        'area_code_area_code_415': area_code_area_code_415,
        'area_code_area_code_510': area_code_area_code_510,
        'international_plan_yes': international_plan_yes
    }])

    st.write("### Giá trị các cột col_1 đến col_7 (HashingEncoder output):")
    for i in range(8):  # từ 0 đến 7
        st.write(f"col_{i} =", input_df[f'col_{i}'].values[0])

    scale_columns = [
        'account_length', 'number_vmail_messages', 'total_day_minutes',
        'total_day_calls', 'total_eve_minutes', 'total_eve_calls',
        'total_night_minutes', 'total_night_calls', 'total_intl_minutes',
        'total_intl_calls', 'number_customer_service_calls'
    ]
    input_df[scale_columns] = scaler.transform(input_df[scale_columns])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ Khách hàng **CÓ THỂ RỜI BỎ** (Churn) với xác suất **{proba:.2%}**")
    else:
        st.success(f"✅ Khách hàng **KHÔNG RỜI BỎ** với xác suất **{1 - proba:.2%}**")
