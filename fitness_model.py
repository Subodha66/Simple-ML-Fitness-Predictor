import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Your Fitness Data (Minutes vs Calories)
data = {
    "Minutes": [10, 20, 30, 40, 50, 60],
    "Calories": [80, 170, 260, 350, 440, 530]
}

df = pd.DataFrame(data)

# 2. Setup the Model
X = df[["Minutes"]]
y = df["Calories"]
model = LinearRegression()
model.fit(X, y)

# 3. Predict for a 45-minute workout
prediction = model.predict([[45]])
print(f"Predicted Calories for 45 mins: {prediction[0]:.2f}")

# 4. Make a Graph (This makes your article look AMAZING)
plt.scatter(X, y, color='blue') # Real data points
plt.plot(X, model.predict(X), color='red') # The "ML Line"
plt.title('ML: Minutes vs Calories')
plt.xlabel('Minutes')
plt.ylabel('Calories')
plt.show()
