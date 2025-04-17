import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
model = joblib.load("decision_tree_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Load test data and labels
test_df = pd.read_csv("test.csv")
true_labels = pd.read_csv("gender_submission.csv")

# Preprocess test data
def preprocess(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    for col in ['Sex', 'Embarked']:
        le = encoders[col]
        df[col] = le.transform(df[col].fillna(le.classes_[0]))
    return df

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Evaluate model
def evaluate_model():
    df = preprocess(test_df.copy())
    X_test = df[features]
    y_pred = model.predict(X_test)
    y_true = true_labels.set_index('PassengerId').loc[test_df['PassengerId']]['Survived'].values
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm, y_pred, y_true

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")

menu = st.sidebar.radio("Navigation", ["📈 Model Evaluation", "🧪 Try a Prediction"])

if menu == "📈 Model Evaluation":
    st.header("📊 Model Evaluation")
    acc, cm, y_pred, y_true = evaluate_model()
    st.write(f"**Accuracy:** `{acc:.2%}`")

    st.subheader("Confusion Matrix")
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig_cm)

    st.subheader("Decision Tree Visualization")
    fig_tree, ax2 = plt.subplots(figsize=(16, 8))
    plot_tree(model, feature_names=features, class_names=["Not Survived", "Survived"], filled=True, ax=ax2)
    st.pyplot(fig_tree)

elif menu == "🧪 Try a Prediction":
    st.header("🧍 Passenger Prediction Form")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0.42, 80.0, 30.0)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
    fare = st.slider("Fare", 0.0, 600.0, 50.0)
    embarked_display = {
        "Cherbourg (C)": "C",
        "Queenstown (Q)": "Q",
        "Southampton (S)": "S"
    }
    embarked_full = st.selectbox("Port of Embarkation", list(embarked_display.keys()))
    embarked = embarked_display[embarked_full]

    if st.button("Predict Survival"):
        # Prepare single input
        input_dict = {
            'Pclass': [pclass],
            'Sex': [encoders['Sex'].transform([sex])[0]],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [encoders['Embarked'].transform([embarked])[0]],
        }
        input_df = pd.DataFrame(input_dict)

        prediction = model.predict(input_df)[0]
        outcome = "🎉 Survived" if prediction == 1 else "💀 Did Not Survive"
        st.success(f"Prediction: **{outcome}**")
