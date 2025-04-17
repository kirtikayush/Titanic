import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

test_df = pd.read_csv("test.csv")
true_labels = pd.read_csv("gender_submission.csv")

clf = joblib.load("decision_tree_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

for col in ['Sex', 'Embarked']:
    le = label_encoders[col]
    test_df[col] = le.transform(test_df[col].fillna(le.classes_[0]))

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_test = test_df[features]
y_pred = clf.predict(X_test)

true_y = true_labels.set_index('PassengerId').loc[test_df['PassengerId']]['Survived'].values

print("✅ Evaluation on Test Data:")
print("Accuracy:", accuracy_score(true_y, y_pred))
print("\nClassification Report:\n", classification_report(true_y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(true_y, y_pred))

sns.heatmap(confusion_matrix(true_y, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(16, 10))
plot_tree(clf, feature_names=features, class_names=["Not Survived", "Survived"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
