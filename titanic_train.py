import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

train_df = pd.read_csv("train.csv")

train_df['Age'].fillna(train_df['Age'].median(),inplace = True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0],inplace = True)

label_encoders = {}
for col in ['Sex','Embarked']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = train_df[features]
y = train_df['Survived']

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X,y)

joblib.dump(clf,"decision_tree_model.pkl")
joblib.dump(label_encoders,"label_encoders.pkl")

print("Model and encoders saved.")