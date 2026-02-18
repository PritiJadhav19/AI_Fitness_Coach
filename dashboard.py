import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Fitness Coach Dashboard", layout="wide")
st.title("🏋️ AI Fitness Coach Dashboard")
st.caption("Loads workout_history.csv and shows your workout stats.")

CSV_FILE = "workout_history.csv"

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

try:
    df = load_data(CSV_FILE)
except Exception as e:
    st.error(f"Could not read {CSV_FILE}. Run your AI coach first to generate it.\n\nError: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
exercise_options = ["All"] + sorted(df["exercise"].unique().tolist())
exercise = st.sidebar.selectbox("Exercise", exercise_options)

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date))

filtered = df.copy()
if exercise != "All":
    filtered = filtered[filtered["exercise"] == exercise]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["timestamp"].dt.date >= start) & (filtered["timestamp"].dt.date <= end)]

# KPIs
total_reps = int(filtered["reps"].sum()) if not filtered.empty else 0
avg_score = float(filtered["score"].mean()) if not filtered.empty else 0.0
total_sets = int(filtered.shape[0])

c1, c2, c3 = st.columns(3)
c1.metric("Total Reps", total_reps)
c2.metric("Avg Score", f"{avg_score:.1f}")
c3.metric("Total Sets Logged", total_sets)

st.divider()

# Table
st.subheader("📋 Workout History")
st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True)

st.divider()

# Charts
st.subheader("📈 Charts")

if filtered.empty:
    st.info("No data for selected filters.")
    st.stop()

# Reps over time
st.write("### Reps Over Time")
fig1 = plt.figure()
plt.plot(filtered["timestamp"], filtered["reps"], marker="o")
plt.xlabel("Time")
plt.ylabel("Reps")
plt.xticks(rotation=20)
st.pyplot(fig1)

# Score over time
st.write("### Score Over Time")
fig2 = plt.figure()
plt.plot(filtered["timestamp"], filtered["score"], marker="o")
plt.xlabel("Time")
plt.ylabel("Score")
plt.xticks(rotation=20)
st.pyplot(fig2)

# Reps by exercise (bar)
st.write("### Total Reps by Exercise")
by_ex = df.groupby("exercise")["reps"].sum().sort_values(ascending=False)
fig3 = plt.figure()
plt.bar(by_ex.index, by_ex.values)
plt.xlabel("Exercise")
plt.ylabel("Total Reps")
st.pyplot(fig3)