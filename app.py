import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Predictor")
st.write("A Machine Learning web app to predict academic performance")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("student_data.csv")

X = df[
    ["Hours", "Attendance", "SleepHours", "PreviousScore", "PracticeTime"]
]
y = df["Marks"]

# ---------------- TRAIN / TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- MODEL EVALUATION ----------------
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

# ---------------- USER INPUT ----------------
st.subheader("🧑‍🎓 Enter Student Details")

hours = st.slider("📚 Study Hours (per day)", 0.0, 15.0, 6.0, 0.5)
attendance = st.slider("🏫 Attendance (%)", 50.0, 100.0, 80.0, 1.0)
sleep = st.slider("😴 Sleep Hours", 4.0, 9.0, 7.0, 0.5)
prev_score = st.slider("📄 Previous Exam Score", 30.0, 90.0, 60.0, 1.0)
practice = st.slider("📝 Practice Time (hrs)", 0.0, 5.0, 2.0, 0.5)

# ---------------- PREDICTION ----------------
if st.button("Predict Performance"):
    prediction = model.predict(
        [[hours, attendance, sleep, prev_score, practice]]
    )[0]

    st.write(f"🎯 **Predicted Marks:** {prediction:.2f}")

    # Pass / Fail
    if prediction >= 40:
        st.success("✅ RESULT: PASS")
    else:
        st.error("❌ RESULT: FAIL")

    # Performance level
    if prediction < 40:
        st.warning("📉 Performance: Poor")
    elif prediction < 60:
        st.info("📘 Performance: Average")
    elif prediction < 80:
        st.success("📗 Performance: Good")
    else:
        st.success("🏆 Performance: Excellent")

# ---------------- MODEL PERFORMANCE ----------------
st.subheader("📊 Model Performance")
st.write(f"R² Score: **{accuracy:.2f}**")

# ---------------- GRAPH ----------------
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Marks")
ax.set_ylabel("Predicted Marks")
st.pyplot(fig)

# ---------------- DATA PREVIEW ----------------
st.subheader("📂 Dataset Preview")
st.dataframe(df.head(20))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Deployed ML Web App | Python • Scikit-Learn • Streamlit")
