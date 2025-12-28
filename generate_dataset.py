import pandas as pd
import random

data = []

for _ in range(1500):
    hours = round(random.uniform(0, 15), 1)
    attendance = round(random.uniform(50, 100), 1)
    sleep = round(random.uniform(4, 9), 1)
    prev_score = round(random.uniform(30, 90), 1)
    practice = round(random.uniform(0, 5), 1)

    marks = (
        hours * 3.5 +
        attendance * 0.25 +
        sleep * 2 +
        prev_score * 0.3 +
        practice * 4 +
        random.uniform(-10, 10)
    )

    marks = max(0, min(100, marks))

    data.append([
        hours, attendance, sleep, prev_score, practice, round(marks, 2)
    ])

df = pd.DataFrame(
    data,
    columns=[
        "Hours", "Attendance", "SleepHours",
        "PreviousScore", "PracticeTime", "Marks"
    ]
)

df.to_csv("student_data.csv", index=False)
print("✅ Dataset generated with", len(df), "rows")

