[app.py](https://github.com/user-attachments/files/32945585/app.py)
[dashboard.html](https://github.com/user-attachments/files/32945482/dashboard.html)

# MyFirst_Repository

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Credit Card Fraud Analytics", page_icon="💳", layout="wide")

DATA_FILE = Path("creditcard analysis2.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Time", "Amount", "Class"]).copy()
    df["Class"] = df["Class"].astype(int)
    df["Fraud Status"] = np.where(df["Class"].eq(1), "Fraud", "Legitimate")
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["Elapsed Hour"] = df["Time"] / 3600
    df["Amount Band"] = pd.cut(
        df["Amount"],
        bins=[-0.01, 25, 50, 100, 250, 500, 1000, np.inf],
        labels=["$0–25", "$25–50", "$50–100", "$100–250", "$250–500", "$500–1K", "$1K+"],
        include_lowest=True
    )
    return df

st.title("💳 Credit Card Fraud Analytics Dashboard")
st.caption("Data Analyst portfolio project • Observed fraud analysis • Interactive business monitoring")

if not DATA_FILE.exists():
    st.error("Place 'creditcard analysis2.csv' in the same folder as this app.py, then restart the app.")
    st.stop()

df = load_data(DATA_FILE)

# Sidebar filters
st.sidebar.header("🔎 Dashboard Filters")
status = st.sidebar.multiselect(
    "Transaction status",
    ["Legitimate", "Fraud"],
    default=["Legitimate", "Fraud"]
)
amount_min, amount_max = float(df.Amount.min()), float(df.Amount.max())
amount_range = st.sidebar.slider(
    "Transaction amount",
    min_value=amount_min, max_value=amount_max,
    value=(amount_min, min(1000.0, amount_max)),
    step=1.0
)
hours = st.sidebar.slider("Hour of day", 0, 23, (0, 23))
bands = st.sidebar.multiselect(
    "Amount band",
    list(df["Amount Band"].dropna().cat.categories.astype(str)),
    default=list(df["Amount Band"].dropna().cat.categories.astype(str))
)

f = df[
    df["Fraud Status"].isin(status)
    & df["Amount"].between(amount_range[0], amount_range[1])
    & df["Hour"].between(hours[0], hours[1])
    & df["Amount Band"].astype(str).isin(bands)
].copy()

# KPI calculations
txn = len(f)
fraud = int(f["Class"].sum())
fraud_rate = fraud / txn * 100 if txn else 0
fraud_value = f.loc[f.Class.eq(1), "Amount"].sum()
avg_fraud_amt = f.loc[f.Class.eq(1), "Amount"].mean() if fraud else 0
avg_legit_amt = f.loc[f.Class.eq(0), "Amount"].mean() if (f.Class.eq(0).sum()) else 0

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Transactions", f"{txn:,}")
c2.metric("Observed Fraud", f"{fraud:,}")
c3.metric("Fraud Rate", f"{fraud_rate:.3f}%")
c4.metric("Fraud Transaction Value", f"${fraud_value:,.2f}")
c5.metric("Avg Fraud Amount", f"${avg_fraud_amt:,.2f}")

st.info(
    "Business use: use this dashboard to identify when fraud is concentrated, which transaction-value bands carry higher observed fraud rates, "
    "and where monitoring or manual review capacity may need attention. The dataset's V1–V28 fields are anonymized model features."
)

# Charts
left, right = st.columns(2)

with left:
    st.subheader("Fraud rate by transaction amount")
    g = f.groupby("Amount Band", observed=False).agg(
        Transactions=("Class","size"),
        Fraud=("Class","sum")
    ).reset_index()
    g["Fraud Rate %"] = np.where(g["Transactions"]>0, g["Fraud"]/g["Transactions"]*100, 0)
    fig = px.bar(g, x="Amount Band", y="Fraud Rate %",
                 hover_data=["Transactions","Fraud"],
                 labels={"Fraud Rate %":"Observed fraud rate (%)"})
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Fraud activity by hour")
    h = f.groupby("Hour").agg(
        Transactions=("Class","size"),
        Fraud=("Class","sum")
    ).reset_index()
    h["Fraud Rate %"] = np.where(h.Transactions>0, h.Fraud/h.Transactions*100, 0)
    fig = px.line(h, x="Hour", y="Fraud Rate %",
                  markers=True, hover_data=["Transactions","Fraud"])
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Transaction volume vs fraud")
    q = f.groupby("Hour").agg(
        Transactions=("Class","size"),
        Fraud=("Class","sum")
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=q.Hour, y=q.Transactions, name="Transactions"))
    fig.add_trace(go.Scatter(x=q.Hour, y=q.Fraud, name="Fraud", mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Transactions"),
        yaxis2=dict(title="Fraud cases", overlaying="y", side="right"),
        height=360, legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Fraud vs legitimate transaction value")
    p = f.groupby("Fraud Status")["Amount"].sum().reset_index()
    fig = px.pie(p, names="Fraud Status", values="Amount", hole=0.55)
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

# Feature comparison
st.subheader("Anonymized feature signals: fraud vs legitimate")
feature_cols = [c for c in df.columns if c.startswith("V")]
means = f.groupby("Fraud Status")[feature_cols].mean().T.reset_index().rename(columns={"index":"Feature"})
if "Fraud" in means.columns and "Legitimate" in means.columns:
    means["Absolute Difference"] = (means["Fraud"] - means["Legitimate"]).abs()
    top = means.sort_values("Absolute Difference", ascending=False).head(10)
    fig = px.bar(
        top.sort_values("Absolute Difference"),
        x="Absolute Difference", y="Feature", orientation="h",
        hover_data=["Fraud","Legitimate"]
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("These V1–V28 variables are anonymized principal-component features, so they should be interpreted as model signals rather than business attributes such as merchant or location.")

# Investigation table
st.subheader("🚨 Fraud investigation queue")
fraud_df = f[f.Class.eq(1)].copy()
show_cols = ["Time","Amount","Hour"] + feature_cols[:8] + ["Class"]
if len(fraud_df):
    fraud_df = fraud_df.sort_values("Amount", ascending=False)
    st.dataframe(fraud_df[show_cols].head(50), use_container_width=True, height=320)
else:
    st.write("No fraud transactions match the current filters.")

# Business insights
st.subheader("📌 Analyst findings & actions")
all_clean = df
overall = all_clean["Class"].mean()*100
band = all_clean.groupby("Amount Band", observed=False).agg(txn=("Class","size"), fraud=("Class","sum"))
band["rate"] = band.fraud / band.txn * 100
top_band = band["rate"].idxmax()
top_band_rate = band.loc[top_band,"rate"]

hour = all_clean.groupby("Hour").agg(txn=("Class","size"), fraud=("Class","sum"))
hour["rate"] = hour.fraud/hour.txn*100
# Avoid tiny denominators for operational insight
eligible = hour[hour.txn >= 100]
peak_hour = int(eligible["rate"].idxmax()) if len(eligible) else int(hour["rate"].idxmax())

a,b,c = st.columns(3)
with a:
    st.markdown("**1. Fraud is rare but operationally important**")
    st.write(f"Observed fraud is {overall:.3f}% of labeled transactions. Because the class is highly imbalanced, raw accuracy alone would be misleading for a fraud model.")
with b:
    st.markdown("**2. Higher-value bands deserve monitoring**")
    st.write(f"The highest observed fraud rate among the predefined amount bands is {top_band} at {top_band_rate:.2f}% in the full dataset.")
with c:
    st.markdown("**3. Time-based monitoring can prioritize review capacity**")
    st.write(f"After requiring at least 100 transactions in an hour, hour {peak_hour:02d}:00 has the highest observed fraud rate in this dataset.")

st.caption(
    "Important: this is descriptive analytics. It does not prove that an amount band or hour causes fraud, and it does not replace a production fraud model. "
    "For deployment, combine this dashboard with precision/recall, false-positive cost, threshold testing, drift monitoring, and confirmed investigation outcomes."
)
