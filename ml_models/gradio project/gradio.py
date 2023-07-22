import gradio as gr
import pandas as pd
import numpy as np
import joblib

# loading model and dataset
path= r'ml_models\gradio project\LR.plk'
model=joblib.load(path)

path_dataset=r'ml_models\gradio project\test_telco.csv'
test= pd.read_csv(path_dataset)

# making prediction
classifier= model.predict(test)

# building GUI for the model
def classify(num):
    if num == 0:
        return "Customer will not Churn"
    else:
        return "Customer will churn"



def predict_churn(SeniorCitizen, Partner, Dependents, tenure, InternetService,
                  OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                  StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                  PaymentMethod, MonthlyCharges, TotalCharges):
    input_data = [
        SeniorCitizen, Partner, Dependents, tenure, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, MonthlyCharges, TotalCharges
    ]

    input_df = pd.DataFrame([input_data], columns=[
        "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ])

    pred = classifier.predict(input_df)
    output = classify(pred[0])

    if output == "Customer will not Churn":
        return [(0, output)]
    else:
        return [(1, output)]

output = gr.outputs.HighlightedText(color_map={
    "Customer will not Churn": "green",
    "Customer will churn": "red"
})

iface = gr.Interface(title= "Customer Churn Prediction For Vodafone PLC",
    fn=predict_churn,
    inputs=[
        gr.inputs.Slider(minimum=0, maximum= 1, step=1, label="SeniorCitizen: Select 1 for Yes and 0 for No"),
        gr.inputs.Dropdown(["Yes", "No"], label="Partner: Do You Have a Partner?"),
        gr.inputs.Dropdown(["Yes", "No"], label="Dependents: Do You Have a Dependent?"),
        gr.inputs.Number(label="tenure: How Long Have You Been with Vodafone in Months?"),
        gr.inputs.Dropdown(["DSL", "Fiber optic", "No"], label="InternetService"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="OnlineSecurity"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="OnlineBackup"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="DeviceProtection"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="TechSupport"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="StreamingTV"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="StreamingMovies"),
        gr.inputs.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.inputs.Dropdown(["Yes", "No"], label="PaperlessBilling"),
        gr.inputs.Dropdown([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="PaymentMethod"),
        gr.inputs.Number(label="MonthlyCharges"),
        gr.inputs.Number(label="TotalCharges")
    ],
    outputs=output,  theme="freddyaboulton/dracula_revamped"
)

iface.launch(share= True )

