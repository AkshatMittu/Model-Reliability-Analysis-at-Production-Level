import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import datetime
import pandas as pd
import joblib
import numpy as np
#from tensorflow.keras.models import load_model
import os
import pickle
import json
from utils import one_hot, create_and_save_embeddings, get_helper_csv_nlp, get_glcm_csv, prep_and_predict, CV_data_quality, get_text_scores_df
from PIL import Image


no_sidebar_style = """
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
"""
st.set_page_config(page_title='Add Model', page_icon='', layout='wide')
st.markdown(no_sidebar_style, unsafe_allow_html=True)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)


model_type = st.sidebar.selectbox("Select type of model to add:", options=['Regression', 'Classification', 'NLP', 'CV'])
st.markdown(f"<p><h2>{model_type}</h2></p>", unsafe_allow_html=True)

st.write("---")

with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/benchmark.json", "r") as af:
    benchmark_metrics = json.load(af)

with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models.json", "r") as m:
    models = json.load(m)

with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/output.json", "r") as o:
    outputs = json.load(o)


def get_probs(data, model, scaler=None):
    
    if "Unnamed: 0" in data.columns:
        data = data.drop('Unnamed: 0', axis=1)    
    if 'target' in data.columns:
        actual_data = data.drop('target', axis=1)
    
    if model_type == "NLP":
        transformed = scaler.transform(data['text'])
        y_probs = model.predict_proba(transformed)
        maxs = np.max(y_probs, axis=1)
        data['probs'] = maxs
        return data
    else:
        one_h = one_hot(actual_data)
        if scaler != None:
            one_h = scaler.transform(one_h)
        y_probs = model.predict_proba(one_h)
        maxs = np.max(y_probs, axis=1)
        data['probs'] = maxs
        return data


def Regression_Classification():
    st.subheader("Upload Model files and Baseline Data")
    model_name = st.text_input("Enter Model Name", placeholder="Eg: Medical Cost Prediction")
    baseline_data, target_name = st.columns(2)
    columns_b = []
    with baseline_data:
        baseline_file = st.file_uploader("Upload Baseline Data:", type=['csv','xlsx'])
        
        if (baseline_file !=  None) and 'csv' in baseline_file.name.split('.'):
            baseline = pd.read_csv(baseline_file)
            columns_b = baseline.columns
            
        if (baseline_file !=  None) and 'xlsx' in baseline_file.name.split('.'):
            baseline = pd.read_excel(baseline_file)
            columns_b = baseline.columns
            
    with target_name:
        target = st.selectbox("Select Target Label (Y) for the model:", options=columns_b)
    
    
    scaler_col, model_col = st.columns(2)
    
    with scaler_col:
        scaler_file = st.file_uploader("Upload Scaler if used:", type=['joblib']) 
    
    with model_col:
        model_file = st.file_uploader("Upload Model:", type=['joblib', 'pkl', 'h5'])
    
        # baseline = baseline.rename(columns={target:'target'})
        # st.dataframe(baseline)
    
    st.subheader("Enter Benchmark metrics:")
    
    if model_type == "Regression":
        
        r2_bench, rmse_bench, mae_bench, mape_bench = st.columns(4)
        with r2_bench:
            r2 = st.number_input("R2 Score")
        with rmse_bench:
            rmse = st.number_input("Root Mean Squared Error")
        with mae_bench:
            mae = st.number_input("Mean Absolute Error")
        with mape_bench:
            mape = st.number_input("Mean Percentage Error")
        
        new_bench = {"R2 Score": r2, "Root Mean Squared Error": rmse, "Mean Absolute Error": mae, "Mean Absolute Percentage Error": mape}
    
    if model_type == "Classification":
        
        f1_bench, precision_bench, recall_bench, roc_bench, accuracy_bench, fpr_bench = st.columns(6)
        with f1_bench:
            f1 = st.number_input("F1 Score")
        with precision_bench:
            precision = st.number_input("Precision")
        with recall_bench:
            recall = st.number_input("Recall")
        with roc_bench:
            roc = st.number_input("ROC")
        with accuracy_bench:
            accuracy = st.number_input("Accuracy")
        with fpr_bench:
            fpr = st.number_input("False Positive Rate")
        
        new_bench = {"Accuracy": accuracy, "F1": f1, "Recall": recall, "Precision": precision, "ROC_AUC": roc, "False Positive Rate": fpr}
            
    if st.button("Submit"):
        
        os.mkdir(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}')
        baseline = baseline.rename(columns={target:'target'})
        baseline.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/baseline.csv")
        
        if scaler_file != None:
            scaler = joblib.load(scaler_file)
            joblib.dump(scaler, f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/scaler.joblib")
        
        if 'pkl' in model_file.name.split('.'):
            model = pickle.load(model_file)
            pickle.dump(model, f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.pkl")
        if 'joblib' in model_file.name.split('.'):
            model = joblib.load(model_file)
            joblib.dump(model, f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.joblib")
        if 'h5' in model_file.name.split('.'):
            model = load_model(model_file)
            model.save(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.h5")
            
        benchmark_metrics[model_name] = new_bench
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/benchmark.json", "w") as f:
            json.dump(benchmark_metrics, f)
        
        models[model_type].append(model_name)
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models.json", "w") as mf:
            json.dump(models, mf)
        
        os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Ground Truths')
        os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Production Runs')

        outputs[model_name] = model_type
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/output.json", "w") as f:
            json.dump(outputs, f)
        
        
        
        
    st.write("---")
    columns_p = []
    st.subheader("Upload Production Data")
    model_name = st.selectbox("Select Model Name", options=models[model_type])
    
    upload_gt, upload_p = st.columns(2)
    with upload_gt:
        gt_file = st.file_uploader("Upload Ground Truth file:", type=["csv",'xlsx'])
        
        if (gt_file !=  None) and 'csv' in gt_file.name.split('.'):
            gt = pd.read_csv(gt_file)
            columns_p = gt.columns
            
        if (gt_file !=  None) and 'xlsx' in gt_file.name.split('.'):
            gt = pd.read_excel(gt_file)
            columns_p = gt.columns
            
            
    with upload_p:
        prod_file = st.file_uploader("Upload Production file:", type=["csv",'xlsx'])
        
        if (prod_file !=  None) and 'csv' in prod_file.name.split('.'):
            prod = pd.read_csv(prod_file)
            
        if (prod_file !=  None) and 'xlsx' in prod_file.name.split('.'):
            prod = pd.read_excel(prod_file)
            
    
    date_p, target_col = st.columns(2)
    with date_p:
        date = st.date_input("Enter Date of production run", datetime.date(2023, 12, 12))
    with target_col:
        p_target = st.selectbox("Select Target Label (Y) for the model:", options=columns_p, key="prod")
    
    
    if st.button("Upload run"):
        
        
        gt = gt.rename(columns={p_target:'target'})
        prod = prod.rename(columns={p_target:'target'})
        
        
        if model_type == "Classification":
            
            dirs = os.listdir(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}")
            
            if 'scaler.joblib' in dirs:
                scaler = joblib.load(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/scaler.joblib")
            else:
                scaler = None
            if 'model.joblib' in dirs:
                model = joblib.load(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.joblib")
            if 'model.pkl' in dirs:
                model = pickle.load(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.pkl")
            if 'model.h5' in dirs:
                model = load_model(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.h5")
            
            prod = get_probs(prod, model, scaler)
        
        gt.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Ground Truths/{date}.csv")
        prod.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Production Runs/{date}.csv")       


def NLP():
    
    st.subheader("Upload Model files and Baseline Data")
    
    model_name_col, output_type_col = st.columns(2)
    
    with model_name_col:
        model_name = st.text_input("Enter Model Name", placeholder="Eg: Medical Cost Prediction", key="NLP_name")
    with output_type_col:
        output = st.selectbox("Select task:", options=['Classification', 'Text'])
    baseline_data, text_name = st.columns(2)
    columns_b = []
    with baseline_data:
        baseline_file = st.file_uploader("Upload Baseline Data:", type=['csv','xlsx'], key="NLP_baseline")
        
        if (baseline_file !=  None) and 'csv' in baseline_file.name.split('.'):
            baseline = pd.read_csv(baseline_file)
            columns_b = baseline.columns
            
        if (baseline_file !=  None) and 'xlsx' in baseline_file.name.split('.'):
            baseline = pd.read_excel(baseline_file)
            columns_b = baseline.columns
            
    with text_name:
        text = st.selectbox("Select Text Column for the model:", options=columns_b, key="NLP_text")
    
    if output == "Text":
        target = st.selectbox("Select Target Label (Y) for the model:", options=columns_b,  key="NLP_target")
    else:
        p_target_col, model_col = st.columns(2)
        
        with p_target_col:
            vectorizer_file = st.file_uploader("Upload vectorizer:", type=['joblib'])
            target = st.selectbox("Select Target Label (Y) for the model:", options=columns_b,  key="NLP_target")
        
        with model_col:
            model_file = st.file_uploader("Upload Model:", type=['joblib', 'pkl', 'h5'], key="NLP_model")
    
    
    st.subheader("Enter Benchmark Metrics:")
    if output == "Classification":
        
        f1_bench, precision_bench, recall_bench, roc_bench, accuracy_bench, fpr_bench = st.columns(6)
        with f1_bench:
            f1 = st.number_input("F1 Score")
        with precision_bench:
            precision = st.number_input("Precision")
        with recall_bench:
            recall = st.number_input("Recall")
        with roc_bench:
            roc = st.number_input("ROC")
        with accuracy_bench:
            accuracy = st.number_input("Accuracy")
        with fpr_bench:
            fpr = st.number_input("False Positive Rate")
        
        new_bench = {"Accuracy": accuracy, "F1": f1, "Recall": recall, "Precision": precision, "ROC_AUC": roc, "False Positive Rate": fpr}
    
    if output == "Text":
        
        f1_bench, precision_bench, recall_bench, bleu_bench, sim_bench = st.columns(5)
        
        with f1_bench:
            f1 = st.number_input("F1 Score")
        with precision_bench:
            precision = st.number_input("Precision")
        with recall_bench:
            recall = st.number_input("Recall")
        with bleu_bench:
            bleu = st.number_input("BLEU")
        with sim_bench:
            sim = st.number_input("Similarity") 

        new_bench = {"Similarity": sim, "F1": f1, "Recall": recall, "Precision": precision, "BLEU": bleu}
        
    if st.button("Submit", key="NLP_button"):
        
        os.mkdir(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}')
        baseline = baseline.rename(columns={target:'target', text:'text'})
        baseline.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/baseline.csv")
        
        if output != "Text":
            
            if vectorizer_file != None:
                vectorizer = joblib.load(vectorizer_file)
                joblib.dump(vectorizer, f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/vectorizer.joblib")
            if 'pkl' in model_file.name.split('.'):
                model = pickle.load(model_file)
                pickle.dump(model, f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.pkl")
            if 'joblib' in model_file.name.split('.'):
                model = joblib.load(model_file)
                joblib.dump(model, f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.joblib")
            if 'h5' in model_file.name.split('.'):
                model = load_model(model_file)
                model.save(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.h5")
            
        benchmark_metrics[model_name] = new_bench
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/benchmark.json", "w") as f:
            json.dump(benchmark_metrics, f)
        
        models[model_type].append(model_name)
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models.json", "w") as mf:
            json.dump(models, mf)
        
        os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Ground Truths')
        os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Production Runs')
        os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds')
        os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/helper')
        
        if output == "Text":
            os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Ground Truths')
            os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/helper_target/Ground Truths')
            os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Production Runs')
            os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/helper_target/Production Runs')
        
        
        create_and_save_embeddings(baseline, 'text', f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds/baseline')
        helper_df = get_helper_csv_nlp(baseline)
        helper_df.to_csv(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/helper/baseline.csv')
        
        outputs[model_name] = output
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/output.json", "w") as f:
            json.dump(outputs, f)

    
    st.write("---")
    columns_p = []
    st.subheader("Upload Production Data")
    
    name_, date_p = st.columns(2)
    with name_:
        model_name = st.selectbox("Select Model Name", options=models[model_type])
    with date_p:
        date = st.date_input("Enter Date of production run", datetime.date(2023, 12, 12))
    
    output = outputs[model_name]
    
    upload_gt, upload_p = st.columns(2)
    with upload_gt:
        gt_file = st.file_uploader("Upload Ground Truth file:", type=["csv",'xlsx'])
        
        if (gt_file !=  None) and 'csv' in gt_file.name.split('.'):
            gt = pd.read_csv(gt_file)
            columns_p = gt.columns
            
        if (gt_file !=  None) and 'xlsx' in gt_file.name.split('.'):
            gt = pd.read_excel(gt_file)
            columns_p = gt.columns
            
            
    with upload_p:
        prod_file = st.file_uploader("Upload Production file:", type=["csv",'xlsx'])
        
        if (prod_file !=  None) and 'csv' in prod_file.name.split('.'):
            prod = pd.read_csv(prod_file)
            
        if (prod_file !=  None) and 'xlsx' in prod_file.name.split('.'):
            prod = pd.read_excel(prod_file)
            
    
    text_name, target_col = st.columns(2)
    with text_name:
        text_col = st.selectbox("Select Text column",options=columns_p, key="textprod")
    with target_col:
        p_target = st.selectbox("Select Target Label (Y) for the model:", options=columns_p, key="prod")
    
    
    if st.button("Upload run"):
        
        
        gt = gt.rename(columns={p_target:'target', text_col:'text'})
        prod = prod.rename(columns={p_target:'target', text_col:'text'})
    
            
        dirs = os.listdir(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}")
        
        if output != "Text":
            if 'vectorizer.joblib' in dirs:
                vectorizer = joblib.load(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/vectorizer.joblib")
            else:
                vectorizer = None

            if 'model.joblib' in dirs:
                model = joblib.load(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.joblib")
            if 'model.pkl' in dirs:
                model = pickle.load(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.pkl")
            if 'model.h5' in dirs:
                model = load_model(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.h5")
    
            prod = get_probs(prod, model, vectorizer)
        
            gt.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Ground Truths/{date}.csv")
            prod.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Production Runs/{date}.csv")
            
            create_and_save_embeddings(prod, 'text', f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds/{date}')
            helper_df = get_helper_csv_nlp(prod)
            helper_df.to_csv(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/helper/{date}.csv')
        
        if output == "Text":
            
            gt.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Ground Truths/{date}.csv")
            prod.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/Production Runs/{date}.csv")
            
            create_and_save_embeddings(prod, 'text', f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds/{date}')
            helper_df_text = get_helper_csv_nlp(prod)
            helper_df_text.to_csv(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/helper/{date}.csv')
            
            create_and_save_embeddings(gt, 'target', f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Ground Truths/{date}')
            helper_df_target_gt = get_helper_csv_nlp(gt, 'target', f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds/{date}.csv', f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Ground Truths/{date}.csv')
            
            helper_df_target_gt.to_csv(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/helper_target/Ground Truths/{date}.csv')
            
            create_and_save_embeddings(prod, 'target', f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Production Runs/{date}')
            helper_df = get_helper_csv_nlp(prod, 'target')
            
            scores_df = get_text_scores_df(gt, prod, pd.read_csv(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Ground Truths/{date}.csv'), pd.read_csv(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Production Runs/{date}.csv'), pd.read_csv(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds/{date}.csv'))
            
            combined_df = pd.concat([helper_df, scores_df], axis=1)
            combined_df.to_csv(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/helper_target/Production Runs/{date}.csv')
    

def CV():

    model_name, label_name = st.columns(2)

    with model_name:
        name = st.text_input("Enter Model Name:", placeholder="X-Ray Classification")

    with label_name:
        label = st.text_input("Enter class name:")

    st.write("Enter Benchmark Metrics:")
    f1_bench, precision_bench, recall_bench, roc_bench, accuracy_bench, fpr_bench = st.columns(6)

    with f1_bench:
        f1 = st.number_input("F1 Score")
    with precision_bench:
        precision = st.number_input("Precision")
    with recall_bench:
        recall = st.number_input("Recall")
    with roc_bench:
        roc = st.number_input("ROC")
    with accuracy_bench:
        accuracy = st.number_input("Accuracy")
    with fpr_bench:
        fpr = st.number_input("False Positive Rate")
    
    new_bench = {"Accuracy": accuracy, "F1": f1, "Recall": recall, "Precision": precision, "ROC_AUC": roc, "False Positive Rate": fpr}

    baseline_dir = f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Baseline/"

    upload_image, model_ = st.columns(2)
    with upload_image:
        with st.form("my-form", clear_on_submit=True):
            baseline_images = st.file_uploader("Upload baseline Images", type=['jpg','jpeg'], accept_multiple_files=True, key=f'Baseline')
            if st.form_submit_button("Upload Images"):
                os.makedirs(f"{baseline_dir}/{label}")
                for i in baseline_images:
                    image = Image.open(i)
                    image.save(f"{baseline_dir}/{label}/{i.name}")
    # with model_:
    #     model_cv = st.file_uploader("Upload Model:", type=['pkl', 'h5'])
    
    st.write("---")
    st.write("Please click on submit after you are done uploading the images")


    if st.button("Submit"):
        
        baseline_paths, baseline_labels = [], []
        baseline_lists = os.listdir(baseline_dir)

        for i in baseline_lists:
            for j in os.listdir(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Baseline/{i}"):
                baseline_paths.append(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Baseline/{i}/{j}")
                baseline_labels.append(i)

        baseline_paths_df = pd.DataFrame({"paths": baseline_paths,
                                        "target": baseline_labels})
        
        baseline_paths_df.to_csv(f"{baseline_dir}/baseline_paths_df.csv", index=False)

        baseline_glcm = get_glcm_csv(baseline_paths_df, [2], [np.pi/2])
        baseline_glcm.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/baseline.csv")

        os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Ground Truths')
        os.makedirs(f'C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Production Runs')

        # if 'pkl' in model_cv.name.split('.'):
        #     model = pickle.load(model_cv)
        #     pickle.dump(model, f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.pkl")
        # if 'h5' in model_cv.name.split('.'):
        #     st.write(model_cv)
        #     model = load_model(model_cv)
        #     model.save(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.h5")
        
        benchmark_metrics[name] = new_bench
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/benchmark.json", "w") as f:
            json.dump(benchmark_metrics, f)
        
        models[model_type].append(name)
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models.json", "w") as mf:
            json.dump(models, mf)

        outputs[name] = "Classification"
        with open("C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/output.json", "w") as f:
            json.dump(outputs, f)
    
    st.write("---")
    st.write("Upload Production Data")
    model_name, label_name = st.columns(2)

    with model_name:
        name = st.selectbox("Select Model Name:", options=models[model_type])

    with label_name:
        label = st.text_input("Enter class name:", key="prod")
    

    date_cv, upload_image_p = st.columns(2)
    with date_cv:
        date = st.date_input("Enter Date of production run", datetime.date(2023, 12, 12))

    production_dir = f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Production/{date}"

    with upload_image_p:
        with st.form("my-form-p", clear_on_submit=True):
            production_images = st.file_uploader("Upload Production Images", type=['jpg','jpeg'], accept_multiple_files=True, key=f'Production')
            if st.form_submit_button("Upload Images"):
                os.makedirs(f"{production_dir}/{label}")
                for i in production_images:
                    image = Image.open(i)
                    image.save(f"{production_dir}/{label}/{i.name}")
    
    st.write("---")
    st.write("Please click on submit after you are done uploading the images")


    if st.button("Submit", key="CV_prod"):

        production_paths, production_labels = [], []
        production_lists = os.listdir(production_dir)

        for i in production_lists:
            for j in os.listdir(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Production/{date}/{i}"):
                production_paths.append(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Production/{date}/{i}/{j}")
                production_labels.append(i)

        production_paths_df = pd.DataFrame({"paths": production_paths,
                                        "target": production_labels})
        
        production_paths_df = CV_data_quality(production_paths_df)
        
        production_paths_df.to_csv(f"{production_dir}/production_paths_df.csv", index=False)

        date_glcm = get_glcm_csv(production_paths_df, [2], [np.pi/2])
        date_glcm.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Ground Truths/{date}.csv")

        model = load_model(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/model.h5")
        production_run_csv = prep_and_predict(model, date_glcm.drop("target", axis=1), production_lists)
        production_run_csv.to_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{name}/Production Runs/{date}.csv")
        
        # Determine Ground Truth files and Production Runs


if model_type == "Regression":
    Regression_Classification()
elif model_type == "Classification":
    Regression_Classification()
elif model_type == "NLP":
    NLP()
elif model_type == "CV":
    CV()

try:
    
    st.sidebar.markdown(f"<h2>Welcome, {st.session_state['username']}</h2>",unsafe_allow_html=True)
    st.sidebar.write(f"ID: {st.session_state['id']+1}")
    st.sidebar.write(f"Occupation: {st.session_state['occupation']}")
    st.sidebar.write(f"Email: {st.session_state['email']}")
    logout = st.sidebar.button('Logout')
    if logout and len(st.session_state) != 0:
        st.session_state.clear()
        switch_page("Landing_Page")
    elif logout and len(st.session_state) == 0:
        st.sidebar.error("Please login first")

except:
    switch_page("Landing_Page")

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
