import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/wine.csv")

x = df["alcohol"]
y = df["quality"]

a, b = np.polyfit(x, y, 1)

y_pred = a * x + b

plt.figure(figsize=(8, 6))

plt.scatter(x, y)

plt.plot(x, y_pred)

plt.title("Linear Regression")
plt.xlabel("alcohol")
plt.ylabel("quality")

plt.grid()

plt.savefig("linear_regression.png")

plt.show()