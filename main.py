import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("Data/student-mat.csv", sep=";")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
df["result"] = np.where(df["G3"] >= 10, "Pass", "At Risk")
print(df[["G3", "result"]].head(10))
print(df["result"].value_counts())
x = df[["G1", "G2", "studytime", "failures", "absences"]]
y = df["result"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
cm = confusion_matrix(y_test, y_pred)
print(cm)
recall = recall_score(y_test, y_pred, pos_label = "At Risk")
print(recall)
precision = precision_score(y_test, y_pred, pos_label = "At Risk")
print(precision)
f1 = f1_score(y_test, y_pred, pos_label = "At Risk")
print(f1)
print(classification_report(y_test, y_pred))
rf_model = RandomForestClassifier(random_state = 42)
rf_model.fit(x_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")
y_pred_rf = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(rf_accuracy)
rf_cm = confusion_matrix(y_test, y_pred_rf)
print(rf_cm)
rf_recall = recall_score(y_test, y_pred_rf, pos_label = "At Risk")
print(rf_recall)
rf_precision = precision_score(y_test, y_pred_rf, pos_label = "At Risk")
print(rf_precision)
rf_f1 = f1_score(y_test, y_pred_rf, pos_label = "At Risk")
print(rf_f1)
print(classification_report(y_test, y_pred_rf))