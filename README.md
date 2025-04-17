# 🚢 Titanic Survival Predictor

A Streamlit web app that predicts whether a passenger would have survived the Titanic disaster based on input features. Built using a Decision Tree Classifier trained on the classic Titanic dataset.

---

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
