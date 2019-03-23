#Supervised learning example: Iris classifcation
import seaborn as sns
import matplotlib.pyplot as plt
# loading iris data via seaborn
iris = sns.load_dataset("iris")
#print(iris)
x=iris.values[:,0] # get data of all rows and first column
y=iris.values[:,1] # get data of all rows and second column
plt.scatter(x,y) # scatter plot of x and y


X_iris = iris.drop('species', axis=1)
X_iris.shape
y_iris = iris['species']
y_iris.shape

from sklearn.model_selection  import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)

from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB() # 2. instantiate model
model.fit(Xtrain, ytrain) # 3. fit model to data
y_model = model.predict(Xtest) # 4. predict on new data
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model))