import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.metric-box {
    padding: 20px;
    border-radius: 15px;
    background: #1c1f26;
    text-align: center;
}
.big-font {
    font-size: 32px;
    font-weight: bold;
}
.sub-font {
    font-size: 16px;
    color: #b0b0b0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("student_data.csv")

X = df[[
    "Hours",
    "Attendance",
    "SleepHours",
    "PreviousScore",
    "PracticeTime"
]]
y = df["Marks"]

# ---------------- TRAIN / TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

# ---------------- HEADER ----------------
st.markdown("## 🎓 Student Performance Predictor")
st.markdown(
    "Predict academic performance using **Machine Learning** based on student habits."
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧑‍🎓 Student Inputs")

hours = st.sidebar.slider("📚 Study Hours / Day", 0.0, 15.0, 6.0, 0.5)
attendance = st.sidebar.slider("🏫 Attendance (%)", 50.0, 100.0, 80.0, 1.0)
sleep = st.sidebar.slider("😴 Sleep Hours", 4.0, 9.0, 7.0, 0.5)
prev_score = st.sidebar.slider("📄 Previous Exam Score", 30.0, 90.0, 60.0, 1.0)
practice = st.sidebar.slider("📝 Practice Time (hrs)", 0.0, 5.0, 2.0, 0.5)

predict = st.sidebar.button("🚀 Predict Performance")

# ---------------- MAIN CONTENT ----------------
col1, col2, col3 = st.columns(3)

if predict:
    raw_prediction = model.predict(
        [[hours, attendance, sleep, prev_score, practice]]
    )[0]

    # Clamp prediction between 0 and 100
    prediction = max(0, min(100, raw_prediction))

    # -------- MARKS CARD --------
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="sub-font">Predicted Marks</div>
                <div class="big-font">{prediction:.1f} / 100</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------- PASS / FAIL --------
    with col2:
        result = "PASS ✅" if prediction >= 40 else "FAIL ❌"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="sub-font">Result</div>
                <div class="big-font">{result}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------- PERFORMANCE LEVEL --------
    with col3:
        if prediction < 40:
            level = "Poor 📉"
        elif prediction < 60:
            level = "Average 📘"
        elif prediction < 80:
            level = "Good 📗"
        else:
            level = "Excellent 🏆"

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="sub-font">Performance</div>
                <div class="big-font">{level}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------- PROGRESS BAR --------
    st.markdown("### 📊 Performance Meter")
    st.progress(int(prediction))

st.divider()

# ---------------- MODEL INFO ----------------
left, right = st.columns(2)

with left:
    st.markdown("### 📈 Model Accuracy")
    st.metric(label="R² Score", value=f"{accuracy:.2f}")

with right:
    st.markdown("### 📉 Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual Marks")
    ax.set_ylabel("Predicted Marks")
    st.pyplot(fig)

st.divider()

# ---------------- DATA PREVIEW ----------------
st.markdown("### 📂 Dataset Preview")
st.dataframe(df.sample(15))

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <center>
    Built with ❤️ using <b>Python, Machine Learning & Streamlit</b><br>
    Publicly deployed ML web application
    </center>
    """,
    unsafe_allow_html=True
)
