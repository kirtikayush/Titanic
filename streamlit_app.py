import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and encoders
models = {
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "Logistic Regression": joblib.load("logistic_regression_model.pkl")
}
encoders = joblib.load("label_encoders.pkl")

# Load test data and labels
test_df = pd.read_csv("test.csv")
true_labels = pd.read_csv("gender_submission.csv")

# Preprocess test data
def preprocess(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    for col in ['Sex', 'Embarked']:
        le = encoders[col]
        df[col] = le.transform(df[col].fillna(le.classes_[0]))
    return df

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Evaluation function
def evaluate_model(model):
    df = preprocess(test_df.copy())
    X_test = df[features]
    y_true = true_labels.set_index('PassengerId').loc[df['PassengerId']]['Survived'].values
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm, y_pred, y_true

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")

menu = st.sidebar.radio("Navigation", ["📈 Model Evaluation", "🧪 Try a Prediction"])

if menu == "📈 Model Evaluation":
    st.header("📊 Model Evaluation")

    model_name = st.selectbox("Choose a model", list(models.keys()))
    selected_model = models[model_name]

    acc, cm, y_pred, y_true = evaluate_model(selected_model)
    st.write(f"**Accuracy for {model_name}:** `{acc:.2%}`")

    st.subheader("Confusion Matrix")
    fig_cm, ax = plt.subplots()
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

    if model_name == "Decision Tree":
        st.subheader("Decision Tree Visualization")
        fig_tree, ax2 = plt.subplots(figsize=(16, 8))
        plot_tree(selected_model, feature_names=features, class_names=["Not Survived", "Survived"], filled=True, ax=ax2)
        st.pyplot(fig_tree)

    st.subheader("📋 All Model Accuracies")
    for name, model_obj in models.items():
        model_acc, _, _, _ = evaluate_model(model_obj)
        st.write(f"{name}: `{model_acc:.2%}`")

elif menu == "🧪 Try a Prediction":
    st.header("🧍 Passenger Prediction Form")

    model_name = st.selectbox("Choose a model", list(models.keys()), key="predict_model")
    selected_model = models[model_name]

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0.42, 80.0, 30.0)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
    fare = st.slider("Fare", 0.0, 600.0, 50.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    if st.button("Predict Survival"):
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

        prediction = selected_model.predict(input_df)[0]
        try:
            prob = selected_model.predict_proba(input_df)[0][1]
            st.success(f"Prediction: **{'🎉 Survived' if prediction == 1 else '💀 Did Not Survive'}** (Survival Probability: `{prob:.2%}`)")
        except:
            st.success(f"Prediction: **{'🎉 Survived' if prediction == 1 else '💀 Did Not Survive'}**")
