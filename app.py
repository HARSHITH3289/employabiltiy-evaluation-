import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Try loading the trained scaler and models, otherwise train and save them
try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
   
    with open("perceptron.pkl", "rb") as perceptron_file:
        perceptron = pickle.load(perceptron_file)
    
    with open("logistic_regression.pkl", "rb") as logreg_file:
        logistic_regression = pickle.load(logreg_file)

except FileNotFoundError:
    print("Model or scaler file not found. Training the models now...")
   
    # Load dataset
    df = pd.read_excel("Student-Employability-Datasets.xlsx", sheet_name="Data")
   
    # Feature selection
    X = df.iloc[:, 1:-2].values
    y = (df["CLASS"] == "Employable").astype(int)  # Convert labels to binary (0 or 1)
   
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
   
    # Train Perceptron model
    perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    perceptron.fit(X_train_scaled, y_train)

    # Train Logistic Regression model
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
    logistic_regression.fit(X_train_scaled, y_train)
   
    # Evaluate models
    y_pred_perceptron = perceptron.predict(X_test_scaled)
    y_pred_logreg = logistic_regression.predict(X_test_scaled)

    accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    
    print(f"Perceptron trained with accuracy: {accuracy_perceptron:.2f}")
    print(f"Logistic Regression trained with accuracy: {accuracy_logreg:.2f}")
   
    # Save the scaler and models
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open("perceptron.pkl", "wb") as perceptron_file:
        pickle.dump(perceptron, perceptron_file)

    with open("logistic_regression.pkl", "wb") as logreg_file:
        pickle.dump(logistic_regression, logreg_file)


def predict_employability(name, general_appearance, manner_of_speaking, physical_condition,
                           mental_alertness, self_confidence, ability_to_present_ideas,
                           communication_skills):
    # Create input array
    input_data = np.array([[general_appearance, manner_of_speaking, physical_condition,
                             mental_alertness, self_confidence, ability_to_present_ideas,
                             communication_skills]])
   
    # Scale input data
    input_scaled = scaler.transform(input_data)
   
    # Use Perceptron model for prediction
    prediction = perceptron.predict(input_scaled)
   
    # Convert prediction to label with emoji
    result = "Employable 😊" if prediction[0] == 1 else " Better try next time - Work Hard! 💪"
    return f"{name} is {result}"

# Define input interface
inputs = [
    gr.Textbox(label="Name"),
    gr.Slider(1, 5, step=1, label="General Appearance"),
    gr.Slider(1, 5, step=1, label="Manner of Speaking"),
    gr.Slider(1, 5, step=1, label="Physical Condition"),
    gr.Slider(1, 5, step=1, label="Mental Alertness"),
    gr.Slider(1, 5, step=1, label="Self Confidence"),
    gr.Slider(1, 5, step=1, label="Ability to Present Ideas"),
    gr.Slider(1, 5, step=1, label="Communication Skills"),
]

# Define output
output = gr.Textbox(label="Employability Prediction")

# Define a custom "Get Evaluated" button
submit_button = gr.Button("Get Evaluated")

# Create Gradio interface with custom button
app = gr.Interface(
    fn=predict_employability, 
    inputs=inputs, 
    outputs=output,
    title="Employability Evaluation", 
    description="Enter your details to get evaluated for employability.",
    submit_btn="Evaluate Yourself🤩"
)
# Launch the app
app.launch()
