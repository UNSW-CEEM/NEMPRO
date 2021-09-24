import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causalnex.structure.pytorch import DAGRegressor

x = np.linspace(0.0, 500, 500)
y = np.linspace(0.0, 500, 500)

reg = DAGRegressor(threshold=0.0,
                   alpha=0.0001,
                   beta=0.5,
                   fit_intercept=True,
                   hidden_layer_units=[10],
                   standardize=True
                   )

X = pd.DataFrame({'x': x})
y = pd.Series(y, name="MEDV")

reg.fit(X, y)

print(reg.predict(pd.DataFrame({'x': [0.0, 250.0, 500.0]})))
# [2634.15699576 4040.53486056]

print(reg.predict(pd.DataFrame({'x': [1.0, 10.0]})))
# [2696.09601899 3978.59717296]

Y = reg.predict(X)
fig, axs = plt.subplots(1, 1, tight_layout=True)
axs.scatter(X['x'], y, color='blue')
axs.scatter(X['x'], Y, color='red')
fig.tight_layout()
plt.show()