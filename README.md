# 🚢 Titanic Survival Predictor

A Streamlit web app that predicts whether a passenger would have survived the Titanic disaster based on input features. Built using a Decision Tree Classifier trained on the classic Titanic dataset.

---

## 📁 File Details 

- decision_tree_model.pkl : Decision tree model saved
- logistic_regression_model.pkl : Logistic Regression model saved
- random_forest_model.pkl : Random Forest model saved
- label_encoders.pkl : Label Encoder saved
- streamlit_app.py : Code for the streamlit app to run from (has all the three models inside it)
- titanic_app.py : Older version of streamlit app which only contained Decision Tree Model
- train.csv : Training data of Titanic
- test.csv : Testing data of Titanic
- gender_submission.csv : Actual result of test.csv
- titanic_train.py : Code for training the data using the Decision Tree **only**
- all_model_training : Code for training the data using Decision Tree, Random Forest Model & Logistic Regression
- titanic_test.py : Testing the saved Decision Tree model locally and getting confusion matrix along decision tree visualisation
- requirements.txt : Requirement files for the streamlit app

--

## 📊 Features

- Train a **Decision Tree** on Titanic survival data
- Evaluate model accuracy and visualize the **confusion matrix**
- **Interactive decision tree plot**
- **Real-time survival prediction** form based on user input

---

## 🧠 Model Details

- Algorithm: `DecisionTreeClassifier` from scikit-learn
- Features used:
  - Passenger class (`Pclass`)
  - Sex
  - Age
  - Siblings/Spouses aboard (`SibSp`)
  - Parents/Children aboard (`Parch`)
  - Fare
  - Port of Embarkation (`Embarked`)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/kirtikayush/titanic.git
cd titanic

# Install dependencies
pip install -r requirements.txt
