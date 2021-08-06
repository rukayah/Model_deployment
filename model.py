from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn import tree
import pickle

data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='micro')}")

with open("classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

# TO LOAD MODEL FROM FILE
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

classifier