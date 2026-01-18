# ============================ IMPORT LIBRARIES ============================
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# ============================ PAGE CONFIG ============================
st.set_page_config(
    page_title="CO₂ Emissions by Vehicle",
    page_icon="🌱",
    layout="wide"
)

# ============================ ECO GREEN THEME ============================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #064e3b, #022c22);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #064e3b;
        font-weight: 700;
    }
    p, span, label {
        color: #065f46;
        font-size: 15px;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #ecfdf5 !important;
    }
    div[data-baseweb="select"] > div,
    input {
        background-color: #ecfdf5 !important;
        border-radius: 8px !important;
        border: 1px solid #10b981 !important;
    }
    button {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
    }
    button:hover {
        background: linear-gradient(90deg, #059669, #047857) !important;
    }
    .block-container {
        padding: 2rem;
        background-color: rgba(255,255,255,0.9);
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================ SIDEBAR ============================
with st.sidebar:
    st.markdown("## 🌍 CO₂ Emissions")
    menu = st.selectbox("Select Section", ("Visualization", "Model"))
    st.markdown("---")
    st.markdown("🌱 *Driving Data Toward a Greener Future*")

# ============================ LOAD DATA ============================
df = pd.read_csv("co2 Emissions.csv")

# ============================ DATA CLEANING ============================
fuel_type_mapping = {
    "Z": "Premium Gasoline",
    "X": "Regular Gasoline",
    "D": "Diesel",
    "E": "Ethanol(E85)",
    "N": "Natural Gas"
}
df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)

# Remove Natural Gas
df_natural = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)

# Remove Outliers
df_features = df_natural[
    ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']
]
df_model = df_features[(np.abs(stats.zscore(df_features)) < 1.9).all(axis=1)]

# ============================ HELPER FUNCTION ============================
def bar_plot(data, x, y, title, rotate=75):
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.barplot(data=data, x=x, y=y, ax=ax, palette="Greens_r")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis='x', rotation=rotate)
    ax.bar_label(ax.containers[0], fontsize=8)
    st.pyplot(fig)

# ============================ VISUALIZATION ============================
if menu == "Visualization":

    st.title("🌱 CO₂ Emissions by Vehicle")
    st.subheader("📊 Dataset Overview")
    st.dataframe(df, use_container_width=True)

    st.subheader("🚗 Brands of Cars")
    bar_plot(df['Make'].value_counts().reset_index(name="Count"),
             "Make", "Count", "Cars by Brand")

    st.subheader("🚘 Top 25 Models")
    bar_plot(df['Model'].value_counts().reset_index(name="Count").head(25),
             "Model", "Count", "Top 25 Car Models")

    st.subheader("🚙 Vehicle Class")
    bar_plot(df['Vehicle Class'].value_counts().reset_index(name="Count"),
             "Vehicle Class", "Count", "Vehicle Classes")

    st.subheader("⚙️ Engine Size")
    bar_plot(df['Engine Size(L)'].value_counts().reset_index(name="Count"),
             "Engine Size(L)", "Count", "Engine Sizes", rotate=90)

    st.subheader("🛢 Fuel Type")
    bar_plot(df['Fuel Type'].value_counts().reset_index(name="Count"),
             "Fuel Type", "Count", "Fuel Types")

    st.header("🌍 CO₂ Emissions Analysis")

    for col, title in [
        ("Make", "CO₂ Emissions by Brand"),
        ("Vehicle Class", "CO₂ Emissions by Vehicle Class"),
        ("Fuel Type", "CO₂ Emissions by Fuel Type")
    ]:
        bar_plot(
            df.groupby(col)["CO2 Emissions(g/km)"].mean().sort_values().reset_index(),
            col, "CO2 Emissions(g/km)", title, rotate=90
        )

    st.header("📦 Outlier Analysis")
    fig, ax = plt.subplots(2, 2, figsize=(18, 10))
    for i, feature in enumerate(df_features.columns):
        ax.flat[i].boxplot(df_features[feature])
        ax.flat[i].set_title(feature)
    st.pyplot(fig)

    st.success(f"Before outlier removal: {len(df)} rows")
    st.success(f"After outlier removal: {len(df_model)} rows")

# ============================ MODEL ============================
else:
    st.title("🔮 CO₂ Emission Prediction")

    X = df_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = df_model['CO2 Emissions(g/km)']

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    col1, col2, col3 = st.columns(3)
    with col1:
        engine_size = st.number_input("Engine Size (L)", step=0.1)
    with col2:
        cylinders = st.number_input("Cylinders", min_value=2, max_value=16)
    with col3:
        fuel_consumption = st.number_input("Fuel Consumption (L/100 km)", step=0.1)

    if st.button("🌱 Predict CO₂ Emission"):
        prediction = model.predict([[engine_size, cylinders, fuel_consumption]])
        st.success(f"🌍 Predicted CO₂ Emissions: **{prediction[0]:.2f} g/km**")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#047857;'>🌱 Data-Driven Insights for a Sustainable Future</p>",
        unsafe_allow_html=True
    )
