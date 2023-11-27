import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import pickle

image = Image.open('rakanlogo.png')

col1,col2,col3 = st.columns([0.06,0.4,0.2])
col2.image(image, width=600)

filename = 'ASModelRakan'
model = pickle.load(open(filename, 'rb'))

df = pd.read_csv("cardataset.csv")

with st.sidebar:
    st.subheader("Set your car's specifications below:")

make_model = st.sidebar.selectbox("🚗 Model", (df.make_model.unique().tolist()))
power_kW = st.sidebar.number_input("⚡ Power in kW",min_value=float(df.power_kW.min()), max_value=float(df.power_kW.max()), value=float(df.power_kW.median()), step=5.0)
age = st.sidebar.number_input("👴 Age",min_value=int(df.age.min()), max_value=int(df.age.max()), value=int(df.age.median()), step=1)
mileage = st.sidebar.number_input("🛣️ Mileage",min_value=df.mileage.min(), max_value=df.mileage.max(), value=df.mileage.median(), step=5000.0)
engine_size = st.sidebar.number_input("📦 Engine Size",min_value=df.engine_size.min(), max_value=df.engine_size.max(), value=df.engine_size.median(), step=100.0)
type = st.sidebar.selectbox("🪪 Type", (df.type.unique().tolist()))


my_dict = {"make_model":make_model, "power_kW":power_kW, "age":age, "mileage":mileage, "engine_size":engine_size, "type":type}
df = pd.DataFrame.from_dict([my_dict])


cols = {
    "make_model": "Car Model",
    "power_kW": "Power in kW",
    "mileage": "Mileage",
    "age": "Age",
    "engine_size": "Engine Size",
    "type": "Type"
}

df_show = df.copy()
df_show.rename(columns = cols, inplace = True)
st.write("Your Car's Specifications: \n")
st.dataframe(df_show, use_container_width=True)


if st.button("Predict"):
    pred = model.predict(df)
    col1, col2 = st.columns(2)
    
    if pred[0].astype(int) < 0:
        st.error("Your car belongs in a junkyard (Predicted value is too low).")
    else:
        col1.write("Rakan predicts your car's value as:")
        st.success(f"${pred[0].astype(int)}")