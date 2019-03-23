from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.5)
print(digits.values())
print(X_test)
#print(Y_train)
#print(Y_test)

lr = LogisticRegression()
print(lr.fit(X_train,Y_train))
print(lr.score(X_test,Y_test))

svm = SVC()
print(svm.fit(X_train,Y_train))
print(svm.score(X_test,Y_test))

rf = RandomForestClassifier(n_estimators=40)
print(rf.fit(X_train,Y_train))
print(rf.score(X_test,Y_test))

from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
print(kf)
#for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
#    print(train_index,test_index)

def get_score(model,X_train,X_test,Y_train,Y_test):
    model.fit(X_train,Y_train)
    return model.score(X_test,Y_test)

score_1=[]
score_2=[]
score_3=[]

for train_index, test_index in kf.split(digits.data):
    X_train,X_test,Y_train,Y_test=digits.data[train_index],digits.data[test_index], \
                                    digits.target[train_index],digits.target[test_index]

    score_1.append(get_score(LogisticRegression(),X_train,X_test,Y_train,Y_test))
    score_2.append(get_score(SVC(), X_train, X_test, Y_train, Y_test))
    score_3.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, Y_train, Y_test))

print(np.mean(score_1))
print(np.mean(score_2))
print(np.mean(score_3))

from sklearn.model_selection import cross_val_score
print(cross_val_score(LogisticRegression(),digits.data,digits.target))
print(cross_val_score(SVC(),digits.data,digits.target))
print(cross_val_score(RandomForestClassifier(),digits.data,digits.target))
