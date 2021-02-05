import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

n_samples = 500

U_A = np.random.uniform(0, 1, n_samples)
A = 2 * (U_A > 0.5) - 1  # needs to be +1 or -1

U_X_1 = np.random.normal(0, 1, n_samples)
X_1 = A + U_X_1

U_X_2 = np.random.normal(0, 1, n_samples)
X_2 = 0 + U_X_2

U_X_3 = np.random.normal(0, 1, n_samples)
X_3 = X_1 - X_2 + U_X_3

# h = (1 + np.exp(-X_1 + X_2 - X_3)) ** (-1)
h = (1 + np.exp(5-(X_1 + X_2 + X_3)**2)) ** (-1)
# h = (1 + np.exp(1 - X_1 + X_2 - X_3 + X_1 * X_2 - 0.1 * X_1 * X_3 + 0.01 * X_2 * X_3)) ** (-1)
Y = 2 * (h > 0.5) - 1  # needs to be +1 or -1

print("The following 4 numbers should be roughly equal for a balanced dataset")
print("Class +1:", sum((Y == 1)))
print("Class -1:", sum((Y == -1)))
print("Attribute +1:", sum((A == 1)))
print("Attribute -1:", sum((A == -1)))

synth_data = np.stack([Y, A, X_1, X_2, X_3], axis=1)
synth_data_frame = pd.DataFrame(synth_data)

# sns.pairplot(synth_data_frame)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("X_1")
ax.set_ylabel("X_2")
ax.set_zlabel("X_3")

# label_pos = (Y == 1)
# label_neg = (Y == -1)
# group_pos = (A == 1)
# group_neg = (A == -1)

ax.scatter(X_1[(Y == 1) & (A == 1)], X_2[(Y == 1) & (A == 1)], X_3[(Y == 1) & (A == 1)], color='green', marker='o')
ax.scatter(X_1[(Y == -1) & (A == 1)], X_2[(Y == -1) & (A == 1)], X_3[(Y == -1) & (A == 1)], color='red', marker='o')
ax.scatter(X_1[(Y == 1) & (A == -1)], X_2[(Y == 1) & (A == -1)], X_3[(Y == 1) & (A == -1)], color='blue', marker='x')
ax.scatter(X_1[(Y == -1) & (A == -1)], X_2[(Y == -1) & (A == -1)], X_3[(Y == -1) & (A == -1)], color='orange', marker='x')

plt.show()
