import matplotlib.pyplot as plt
import numpy as np
# 1) choose a model
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
# 2) Choose hypar_parameters
model = LinearRegression(fit_intercept=True)
# 3) Arrange x axis data from two dimension to one dimension
X = x[:, np.newaxis]
# 4) Fit the model to data of y axis
model.fit(X, y)
# find slope and initial data
print(model.coef_)
print(model.intercept_)
# 5) predict the y data based on slope and initial value
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
print(xfit)
print(yfit)
# plotting original data
plt.scatter(x, y)
# plotting modelled data
plt.plot(xfit, yfit);
plt.show()