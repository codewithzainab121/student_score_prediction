# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1️⃣ Load dataset
data = pd.read_csv("student_score_prediction/data.csv")

# 2️⃣ Split features and target
X = data[['Hours']]  # independent variable
y = data['Score']    # dependent variable

# 3️⃣ Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5️⃣ Make predictions
y_pred = model.predict(X_test)

# 6️⃣ Show results
for i, hours in enumerate(X_test['Hours']):
    print(f"Hours studied: {hours}, Actual Score: {y_test.iloc[i]}, Predicted Score: {y_pred[i]:.2f}")

# 7️⃣ Optional: predict score for custom hours
hours_studied = 7
predicted_score = model.predict([[hours_studied]])
print(f"\nIf a student studies {hours_studied} hours, predicted score = {predicted_score[0]:.2f}")
