import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score, average_precision_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import joblib
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
from num2words import num2words
import re, string
import tensorflow_hub as hub
from sklearn.manifold import TSNE
import networkx as nx
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import textstat
from spellchecker import SpellChecker
import random
from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import os
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


def get_labels(baseline_data, model_name):
    if model_name == "Heart Disease Prediction":
        return ['Disease present', 'Disease not present']
    else:
        return list(baseline_data['target'].unique())

def EncodeLabels(gt_label, pd_label):
    
    label = LabelEncoder()
    gt = label.fit_transform(gt_label)
    pd = label.transform(pd_label)
    return gt, pd

def one_hot(data):
    if 'object' in [data[i].dtype for i in data]:
        object_data = data.select_dtypes(include='object')
        object_df = pd.get_dummies(object_data)
        rest_data = data.select_dtypes(exclude='object')
        return pd.concat([object_df, rest_data],axis=1)
    else:
        return data

def regression_metrics(true_labels,predicted_labels):
    metrics = dict()
    metrics['Mean Absolute Error'] = int(round(mean_absolute_error(true_labels, predicted_labels),2))
    # metrics['Mean Squared Error'] = np.round(mean_squared_error(true_labels, predicted_labels), decimals=2)
    metrics['Root Mean Squared Error'] = int(np.round(np.sqrt(float(mean_squared_error(true_labels, predicted_labels))), decimals=2))
    metrics['R2 Score'] = np.round(r2_score(true_labels, predicted_labels), decimals=2)
    metrics['Mean Percentage Error'] = np.round(np.mean((true_labels - predicted_labels) / true_labels) * 100, decimals=2)
    metrics['Mean Absolute Percentage Error'] = np.round(np.mean(np.abs((true_labels - predicted_labels) / true_labels)) * 100, decimals=2)
    return metrics

def classification_metrics(true_labels,predicted_labels, labels=None):
    metrics = dict()
    metrics['report'] = classification_report(true_labels, predicted_labels, output_dict=True, target_names=labels)
    if len(set(true_labels)) > 2 or len(set(true_labels)) > 2: 
        lb = LabelBinarizer()
        true_m = lb.fit_transform(true_labels)
        pred_m = lb.transform(predicted_labels)
        
        metrics['Accuracy'] = np.round(accuracy_score(true_labels, predicted_labels)*100, decimals=2)
        metrics['Precision'] = np.round(precision_score(true_m, pred_m, average="weighted")*100, decimals=2)
        metrics['Recall'] = np.round(recall_score(true_m, pred_m, average="weighted")*100, decimals=2)
        metrics['F1'] = np.round(f1_score(true_m, pred_m, average="weighted")*100, decimals=2)
        metrics['ROC_AUC'] = np.round(roc_auc_score(true_m, pred_m, multi_class="ovr")*100, decimals=2)
        
        cm = confusion_matrix(true_labels, predicted_labels)
        FP = np.sum(cm, axis=0) - np.diag(cm)
        TN = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)

        total_FP = np.sum(FP)
        total_TN = np.sum(TN)
    
        metrics['False Positive Rate'] = np.round((total_FP / (total_FP + total_TN))*100, decimals=2)
        
        return metrics
    
    else:
        
        metrics['Accuracy'] = np.round(accuracy_score(true_labels, predicted_labels)*100, decimals=2)
        metrics['Precision'] = np.round(precision_score(true_labels, predicted_labels)*100, decimals=2)
        metrics['Recall'] = np.round(recall_score(true_labels, predicted_labels)*100, decimals=2)
        metrics['F1'] = np.round(f1_score(true_labels, predicted_labels)*100, decimals=2)
        metrics['ROC_AUC'] = np.round(roc_auc_score(true_labels, predicted_labels)*100, decimals=2)
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(true_labels, predicted_labels).ravel()
        # metrics['True Positive Rate'] = true_positive / (true_positive + false_negative)
        metrics['False Positive Rate'] = np.round((false_positive / (false_positive + true_negative))*100, decimals=2)
        
        return metrics

def plot_reg_target(true_labels, predicted_labels):
    x = np.arange(len(true_labels)) 
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=true_labels,
                    mode='lines+markers',
                    name='Ground Truth'))
    figure.add_trace(go.Scatter(x=x, y=predicted_labels,
                    mode='lines+markers',
                    name='Predicted'))
    return figure

def plot_bar(production,benchmark):

    figure = go.Figure()
    figure.add_trace(go.Bar(name='Production Run', x=list(production.keys()), y=list(production.values()), text=list(production.values())))
    figure.add_trace(go.Bar(name='Benchmark', x=list(benchmark.keys()), y=list(benchmark.values()), text=list(benchmark.values())))
    return figure

def plot_confusion_matrix(true_labels,predicted_labels, class_labels):
    
    cm_df = pd.DataFrame(confusion_matrix(true_labels, predicted_labels), index=class_labels, columns=class_labels)
    fig = px.imshow(cm_df,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=class_labels,
                    y=class_labels,
                    text_auto=True)
    return fig

def plot_roc(true_labels, predicted_labels):
    
    if len(set(true_labels)) > 2 or len(set(true_labels)) > 2: 
        lb = LabelBinarizer()
        true_m = lb.fit_transform(true_labels)
        pred_m = lb.transform(predicted_labels)
        
        
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(len(lb.classes_)):
            fpr[i], tpr[i], _ = roc_curve(true_m[:, i], pred_m[:, i])
            roc_auc[lb.classes_[i]] = round(auc(fpr[i], tpr[i]), 2)

        figure = px.area(x=fpr, y=tpr,
            title=f'ROC Curve (AUC={roc_auc})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)
        return figure, roc_auc
    
    else:
        
        false_positve_r, true_positive_r, thresholds = roc_curve(true_labels, predicted_labels)
        roc_auc = auc(false_positve_r, true_positive_r)
        figure = px.area(x=false_positve_r, y=true_positive_r,
            title=f'ROC Curve (AUC={round(roc_auc, 2)})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)
        return figure, roc_auc

def plot_PR(true_labels, predicted_labels):
    
    if len(set(true_labels)) > 2 or len(set(true_labels)) > 2 : 
        lb = LabelBinarizer()
        true_m = lb.fit_transform(true_labels)
        pred_m = lb.transform(predicted_labels)
    
        precision, recall, average_precision = dict(), dict(), dict()
        
        for i in range(len(lb.classes_)):
            precision[i], recall[i], _ = precision_recall_curve(true_m[:, i], pred_m[:, i])
            average_precision[lb.classes_[i]] = round(average_precision_score(true_m[:, i], pred_m[:, i]), 2)
        
        average = {k:round(i*100,2) for k,i in average_precision.items()}
        
        figure = px.area(x=[i[0] for i in list(recall.values())], y=[i[0] for i in list(precision.values())],
            title=f'Precision: {average}',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0)
        
        return figure, precision, recall

    else:
        
        precision, recall, _ = precision_recall_curve(true_labels, predicted_labels)
        precision_ = np.round(precision_score(true_labels, predicted_labels)*100, decimals=2)
        recall_ = np.round(recall_score(true_labels, predicted_labels)*100, decimals=2)
        figure = px.area(x=recall, y=precision,
            title=f'Precision: {precision}, Recall: {recall}',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0)
        return figure, precision_, recall_

def data_completeness(df):
    completeness = round((len(df.dropna()) / len(df)) * 100, 2)
    missing_data = df.isnull().sum()
    return completeness, len(df), len(df.dropna()), len(df) - len(df.dropna()), missing_data

def donut_for_completeness(value):
    df = {"Label": ["Complete", "Missing"], "Value": [value, 100 - value]}
    df = pd.DataFrame(df)
    figure = px.pie(df, names='Label', values='Value', hole=0.8)
    return figure

def plot_missing_bar(series, len_df):
    df = {"Feature": [i for i in series.index], "Values": [round((i / len_df), 4) * 100  for i in series]}
    df = pd.DataFrame(df)
    figure = px.bar(df, x="Feature", y="Values", text_auto=True)
    return figure

def data_uniqueness(df):
    score = round((len(df.drop_duplicates()) / len(df)) * 100, 2)
    uniqueness_score, uniqueness_value = dict(), dict()
    uniqueness_score["Feature"] = [i for i in df]
    uniqueness_score["Value"] = [round((df[i].nunique() / len(df[i])) * 100, 4) for i in df]
    uniqueness_value["Feature"] = [i for i in df]
    uniqueness_value["Value"] = [len(df[i].unique()) for i in df]
    return score, uniqueness_score, uniqueness_value, len(df), len(df.drop_duplicates())

def donut_for_uniqueness(value):
    df = {"Label": ["Unique", "Duplicate"], "Value": [value, 100 - value]}
    df = pd.DataFrame(df)
    figure = px.pie(df, names='Label', values='Value', hole=0.8)
    return figure

def donut_for_validity(value):
    df = {"Label": ["Valid Data", "Invalid Data"], "Value": [value, 100 - value]}
    df = pd.DataFrame(df)
    figure = px.pie(df, names='Label', values='Value', hole=0.8)
    return figure

def plot_unique_bar(uni_dict):
    figure = px.bar(x=uni_dict["Feature"], y=uni_dict["Value"], text_auto=True)
    return figure

def outliers_for_num_data(data):
    
    outliers_indices = {i:[] for i in data.columns}
    
    for i in data.select_dtypes(exclude='object').columns:
        
        Q1 = data[i].quantile(0.25)
        Q3 = data[i].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[i] < lower_bound) | (data[i] > upper_bound)]
        outliers_indices[i] = outliers.index.tolist()
    
    return outliers_indices
    
def validity_check(baseline_data, production_data):
    
    baseline_data = baseline_data.loc[:, ~baseline_data.columns.str.contains('^Unnamed')]
    common_columns = baseline_data.columns.intersection(production_data.columns)
    production_data = production_data[common_columns]
    
    miss_dict = {i:[] for i in baseline_data.columns}
    cat_has_int = {i:[] for i in baseline_data.select_dtypes(include='object').columns}
    int_has_cat = {i:[] for i in baseline_data.select_dtypes(exclude='object').columns}
    indices = []
    outliers_index = {i:[] for i in baseline_data.columns}
    
    if list(baseline_data.columns) == list(production_data.columns):
        for column in baseline_data.columns:
            
            if production_data[column].isnull().any():
                missing_indices = production_data[production_data[column].isnull()].index.tolist()
                
                miss_dict[column] = missing_indices
            
            if production_data[column].dtype == 'object' and baseline_data[column].dtype == 'object':
                unique_bs = set(baseline_data[column].unique())
                unique_pd = set(production_data[column].unique())
                extras = unique_bs.symmetric_difference(unique_pd)
                
                cat_has_int_id = []
                for i in extras:
                    index_list = production_data[production_data[column]  ==  i].index.tolist()
                    cat_has_int_id.extend(index_list)
                cat_has_int[column] = cat_has_int_id
            
            # if baseline_data[column].dtype != production_data[column].dtype:
            #     int_has_cat_id = []
            #     for index, value in production_data[column].items():
                    
            #         try:
            #             float(value)
            #         except ValueError:
            #             int_has_cat_id.append(index)
            #     int_has_cat[column] = int_has_cat_id
                        
    for k in list(miss_dict.keys()):
        indices.extend(miss_dict[k])
    
    for k in list(cat_has_int.keys()):
        indices.extend(cat_has_int[k])
    
    # for k in list(int_has_cat.keys()):
    #     indices.extend(int_has_cat[k])
    
    num_invalid = len(set(indices))
    total_rows = len(production_data)
    score = np.round((total_rows - num_invalid) / total_rows * 100, 2)
    
    if num_invalid != 0:
        clean_df = production_data.drop(indices, axis=0)
        outliers_index = outliers_for_num_data(clean_df)
    
    return cat_has_int, miss_dict, score, indices, outliers_index, num_invalid

def plot_validity_bar(f_name, missing, wrong, BR):
    figure = go.Figure()
    figure.add_trace(go.Bar(name="Missing", x=[f_name], y=[missing], text=[missing]))
    figure.add_trace(go.Bar(name="Wrong Data Types", x=[f_name], y=[wrong], text=[wrong]))
    figure.add_trace(go.Bar(name="Outliers", x=[f_name], y=[BR], text=[BR])) 
    return figure

def ks_test(baseline, production, num_ft=None, nlp=False, alpha=0.05, b_name="Baseline Data"):
    
    if nlp == True:
        x = "Frequency"
    else:
        x = "Target Values"
    if num_ft == None or nlp == True:
        ks_stat, p_value = ks_2samp(baseline, production)
        results = {'ks-stat': round(ks_stat,2), 'p-value': round(p_value,2), 'status': p_value < alpha}
        
        sample1 = np.sort(baseline)
        sample2 = np.sort(production)
        
        cdf1 = np.arange(1, len(sample1) + 1) / len(sample1)
        cdf2 = np.arange(1, len(sample2) + 1) / len(sample2)
        
        figure = go.Figure() 
        figure.add_trace(go.Scatter(x=sample1, y=cdf1, mode='lines', name=b_name))
        figure.add_trace(go.Scatter(x=sample2, y=cdf2, mode='lines', name="Production Data"))
        figure.update_layout(
                        xaxis_title=x,
                        yaxis_title='Cumulative Distribution ',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=500)
        figure.update_xaxes(tickfont={"size":14, "color":"black"})
        figure.update_yaxes(tickfont={"size":14, "color":"black"})
        
        return results, figure
    else:
        results = dict()
        figures = dict()
        for i in num_ft:
            ks_stat, p_value = ks_2samp(baseline[i], production[i])
            results[i] = {'ks-stat': round(ks_stat,2), 'p-value': round(p_value,2), 'status': p_value < alpha}
            
            sample1 = np.sort(baseline[i])
            sample2 = np.sort(production[i])
            
            cdf1 = np.arange(1, len(sample1) + 1) / len(sample1)
            cdf2 = np.arange(1, len(sample2) + 1) / len(sample2)
            
            figure = go.Figure() 
            figure.add_trace(go.Scatter(x=sample1, y=cdf1, mode='lines', name=b_name))
            figure.add_trace(go.Scatter(x=sample2, y=cdf2, mode='lines', name="Production Data"))
            figure.update_layout(
                            xaxis_title='Feature Values',
                            yaxis_title='Cumulative Distribution ',
                            # title_font={"size": 20},
                            xaxis_title_font={"size":16, "color":"black"},
                            yaxis_title_font={"size":16, "color":"black"},
                            width=1080,
                            height=500)
            figure.update_xaxes(tickfont={"size":14, "color":"black"})
            figure.update_yaxes(tickfont={"size":14, "color":"black"})
            
            figures[i] = figure                 
        return results, figures

def chi_test(baseline, production, categorical_ft=None, alpha=0.05):

    if categorical_ft == None:
        cross_tab = pd.crosstab(baseline, production)
        
        chi_stat, p_value, _, _ = chi2_contingency(cross_tab)
        results = {'chi': round(chi_stat, 2), 'p-value': round(p_value,2), 'status': p_value > alpha}
        
        proportions_baseline, proportions_production = dict(), dict()
        for j in list(baseline.unique()):
            proportions_baseline[j] = round((baseline.value_counts()[j] / len(baseline)) * 100, 2)
            if j in list(production.unique()):
                proportions_production[j] = round((production.value_counts()[j] / len(production)) * 100, 2)
            else:
                proportions_production[j] = 0
    
        figure = go.Figure()
        figure.add_traces(go.Bar(x=list(proportions_baseline.keys()), y=list(proportions_baseline.values()), text=list(proportions_baseline.values()), name="Baseline"))
        figure.add_traces(go.Bar(x=list(proportions_production.keys()), y=list(proportions_production.values()), text=list(proportions_production.values()), name="Production"))
        
        figure.update_layout(
                        xaxis_title=f'Categories in Target',
                        yaxis_title='Category Proportions (%) ',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=500)
        figure.update_xaxes(tickfont={"size":14, "color":"black"})
        figure.update_yaxes(tickfont={"size":14, "color":"black"})
        figure.update_layout(barmode='group')
        
        return results, figure
    
    else:
        results, figures = dict(), dict()

        for i in categorical_ft:
            
            cross_tab = pd.crosstab(baseline[i], production[i])
            
            chi_stat, p_value, _, _ = chi2_contingency(cross_tab)
            results[i] = {'chi': round(chi_stat, 2), 'p-value': round(p_value,2), 'status': p_value > alpha}
            
            proportions_baseline, proportions_production, prop1, prop2 = dict(), dict(), dict(), dict()
            for j in list(baseline[i].unique()):
                prop1[j] = round((baseline[i].value_counts()[j] / len(baseline)) * 100, 2)
                prop2[j] = round((production[i].value_counts()[j] / len(production)) * 100, 2)

            proportions_baseline[i] = prop1
            proportions_production[i] = prop2
        
            figure = go.Figure()
            figure.add_traces(go.Bar(x=list(proportions_baseline[i].keys()), y=list(proportions_baseline[i].values()), text=list(proportions_baseline[i].values()), name="Baseline"))
            figure.add_traces(go.Bar(x=list(proportions_production[i].keys()), y=list(proportions_production[i].values()), text=list(proportions_production[i].values()), name="Production"))
            
            figure.update_layout(
                            xaxis_title=f'Categories in {i}',
                            yaxis_title='Category Proportions (%) ',
                            # title_font={"size": 20},
                            xaxis_title_font={"size":16, "color":"black"},
                            yaxis_title_font={"size":16, "color":"black"},
                            width=1080,
                            height=500)
            figure.update_xaxes(tickfont={"size":14, "color":"black"})
            figure.update_yaxes(tickfont={"size":14, "color":"black"})
            figure.update_layout(barmode='group')
            
            figures[i] = figure

        return results, figures

def determine_dtype_ft(df):
    categorical_ft, numerical_ft = [], []
    for i in df.columns:
        if df[i].dtype == 'object':
            categorical_ft.append(i)
        else:
            numerical_ft.append(i)
    return categorical_ft, numerical_ft

def pred_drift_plots(production, model_type):

    if model_type == "Regression":
        y = 'Statistic Value'
    if model_type == "Classification":
        y = 'Probability Value'
    
    t_max, t_min, t_mean, t_median = dict(), dict(), dict(), dict()
    for i in production:
        t_max[i] = round(max(production[i]),2)
        t_min[i] = round(min(production[i]),2)
        t_mean[i] = round(np.mean(production[i]),2)
        t_median[i] = round(np.median(production[i]),2)
        
    stats = {'Max': t_max, 'Min': t_min, 'Mean': t_mean, 'Median': t_median}

    figures = dict()
    for i in stats:
        figure = go.Figure()
        figure.add_traces(go.Bar(x=list(stats[i].keys()), y=list(stats[i].values()), text=list(stats[i].values())))
        figure.update_layout(title=f'{i} Statistic',
                        xaxis_title=f'Production Runs',
                        yaxis_title=f'{y}',
                        title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=400,
                        height=400)
        figure.update_xaxes(tickfont={"size":14, "color":"black"})
        figure.update_yaxes(tickfont={"size":14, "color":"black"})
        figure.update_layout(barmode='group')
        figures[i] = figure
    return stats, figures

def text_cleaning_with_numbers_as_text(text):
    lemmatize = WordNetLemmatizer()
    text = text.lower()
    contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would",
                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    text = ' '.join([contractions[word] if word in contractions else word for word in text.split()])
    text = re.sub(r'[^0-9a-zA-Z\s]+','',text)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = ' '.join([num2words(word) if word.isdigit() else word for word in text.split()])
    while re.search('-',text):
        text = re.sub('-',' ',text)
    words = nltk.word_tokenize(text)
    stopword = set(stopwords.words('english'))
    new_words = [word for word in words if word not in stopword]
    lem_words = [lemmatize.lemmatize(word) for word in new_words]
    return ' '.join(lem_words)

def load_model(model_name):
    if model_name == "Medical Cost Prediction":
        return joblib.load(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.joblib")

def syntax_drift(results_b, results_pr):
    
    baseline_freq = Counter(results_b)
    production_freq = Counter(results_pr)
    
    baseline_cloud = WordCloud(width=650, height=350, background_color="white").generate(' '.join([i for i in results_b]))
    prod_cloud = WordCloud(width=650, height=350, background_color="white").generate(' '.join([i for i in results_pr]))
    
    top_5_baseline = dict(sorted(baseline_freq.items(), key=lambda a:a[1], reverse=True)[:5])
    top_5_production = dict(sorted(production_freq.items(), key=lambda a:a[1], reverse=True)[:5])
    
    uncommon = set(results_pr) - set(results_b)
    
    uncommon_freq = {i: production_freq[i] for i in uncommon}
    top_5_uncommon = dict(sorted(uncommon_freq.items(), key=lambda a:a[1], reverse=True)[:5])
    
    figure_bvp = go.Figure()
    figure_bvp.add_trace(go.Bar(name='Baseline Words', x=list(top_5_baseline.keys()), y=list(top_5_baseline.values()), text=list(top_5_baseline.values())))
    figure_bvp.add_trace(go.Bar(name='Production Words', x=list(top_5_baseline.keys()), y=[production_freq[i] for i in top_5_baseline], text=[production_freq[i] for i in top_5_baseline]))
    figure_bvp.update_layout(
                        #title=f'Frequencies of Top 5 Baseline words in Production',
                        xaxis_title='Words in Baseline Data',
                        yaxis_title='Count',
                        #title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=450)
    figure_bvp.update_xaxes(tickfont={"size":14, "color":"black"})
    figure_bvp.update_yaxes(tickfont={"size":14, "color":"black"})
    figure_bvp.update_layout(barmode='group')
                         
    figure_pvb = go.Figure()
    figure_pvb.add_trace(go.Bar(name='Production Words', x=list(top_5_production.keys()), y=list(top_5_production.values()), text=list(top_5_production.values())))
    figure_pvb.add_trace(go.Bar(name='Baseline Words', x=list(top_5_production.keys()), y=[baseline_freq[i] for i in top_5_production], text=[baseline_freq[i] for i in top_5_production]))
    figure_pvb.update_layout(
                        #title=f'Frequencies of Top 5 Production words in Baseline',
                        xaxis_title='Words in Production Data',
                        yaxis_title='Count',
                        #title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=450)
    figure_pvb.update_xaxes(tickfont={"size":14, "color":"black"})
    figure_pvb.update_yaxes(tickfont={"size":14, "color":"black"})
    figure_pvb.update_layout(barmode='group')
                         
    figures = dict()
    figures['baseline cloud'] = baseline_cloud
    figures['production cloud'] = prod_cloud
    figures['top 5 baseline bar'] = figure_bvp
    figures['top 5 production bar'] = figure_pvb
    
    return figures, baseline_freq, production_freq, uncommon, uncommon_freq, top_5_uncommon, top_5_baseline, top_5_production

def preprocess_for_embedding(text):
    
    text = text.lower()
    contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would",
                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    text = ' '.join([contractions[word] if word in contractions else word for word in text.split()])
    text = re.sub(r'[^0-9a-zA-Z\s]+','',text)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = ' '.join([num2words(word) if word.isdigit() else word for word in text.split()])
    while re.search('-',text):
        text = re.sub('-',' ',text)
    while re.search(',',text):
        text = re.sub(',',' ',text)
    return text

def creat_embed(text, use):
    embeds = use([text])
    return embeds[0].numpy()

def create_and_save_embeddings(text_df, text_col, file_name):
    
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use = hub.load(module_url)
    
    text_df['embeds'] = text_df[text_col].apply(preprocess_for_embedding) 
    
    embed_df = dict()
    for i in range(len(text_df)):
        embed_df[i] = creat_embed(text_df.iloc[i]['embeds'], use).tolist()
    
    df = pd.DataFrame(embed_df).T
    
    df.to_csv(f"{file_name}.csv")


def semantic_drift(production_run, model_name, col='text', b_name="Baseline"):
    
    if col == 'text':
        baseline = pd.read_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds/baseline.csv")
        production = pd.read_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds/{production_run}.csv")
    elif col == 'target':
        baseline = pd.read_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Ground Truths/{production_run}.csv")
        production = pd.read_csv(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/embeds_target/Production Runs/{production_run}.csv")
        
    if 'Unnamed: 0' in production.columns:
        production.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Unnamed: 0' in baseline.columns:
        baseline.drop('Unnamed: 0', axis=1, inplace=True)
    
    tsne_b = TSNE(n_components = 2, random_state=42)
    tsne_base = tsne_b.fit_transform(baseline[:1000])
    
    tsne_p = TSNE(n_components = 2, random_state=42)
    tsne_pr = tsne_p.fit_transform(production[:1000])
    
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[i[0] for i in tsne_base], y=[i[1] for i in tsne_base],
                    mode='markers',
                    name=b_name))
    figure.add_trace(go.Scatter(x=[i[0] for i in tsne_pr], y=[i[1] for i in tsne_pr],
                    mode='markers',
                    name='Production'))
    figure.update_layout(
                        #xaxis_title=x,
                        #yaxis_title=' ',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=500)
    
    if col == "target":
        return figure
    else:
        edges = []
        pr_node_sizes = dict()
        bs_nodes_used = []
        edge_alpha = dict()
        
        random_state = random.randint(1,100)
        
        baseline_sample = baseline.sample(15, random_state=random_state)
        prod_sample = production.sample(15, random_state=random_state)
        
        for i in range(len(prod_sample)):
            sim = cosine_similarity(X=[prod_sample.iloc[i]], Y=baseline_sample)
            indices = np.where(sim>0.45)[1]
            bs_nodes_used.extend(list(baseline_sample.iloc[indices].index))
            pr_node_sizes[prod_sample.index[i]] = len(indices) + 100
            for j in indices:
                edges.append((prod_sample.index[i], baseline_sample.index[j]))
                edge_alpha[(prod_sample.index[i], baseline_sample.index[j])] = sim[0][j]
            
        bs_node_sizes = {i:bs_nodes_used.count(i) + 100 for i in baseline_sample.index}
        
        isolated = set(prod_sample.index) - set([i for (i,j) in edges]) 
        isolated_score = round(len(isolated) / len(prod_sample), 2) * 100
        
        return figure, isolated_score, isolated, edges, edge_alpha, pr_node_sizes, bs_node_sizes
        
    
def text_cleaning(text):

    text = text.lower()
    contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would",
                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    text = ' '.join([contractions[word] if word in contractions else word for word in text.split()])
    text = re.sub(r'[^0-9a-zA-Z\s]+','',text)
    return text

def get_words(data, col='text'):
    
    cleaned = data[col].apply(text_cleaning)
    words = [i.split() for i in cleaned]
    results = []
    for i in words:
        results.extend(i) 

    return results

def check_spelling(words):
    
    checker = SpellChecker()
    misspelled = checker.unknown(words)
    
    index = random.randint(0,len(misspelled))
    
    return len(misspelled), round(len(set(misspelled)) / len(set(words)), 2) * 100, list(misspelled)[index:index+10]

def nlp_quality_metrics(helper, quality_type, col='text'):
    
    max_index, min_index = helper[quality_type].argmax(), helper[quality_type].argmin()
    max_text, min_text = helper[col].iloc[max_index], helper[col].iloc[min_index]
    
    ma, mi, avg = helper[quality_type].max(), helper[quality_type].min(), round(helper[quality_type].mean(), 2)
    
    return ma, mi, avg, max_text, min_text, max_index, min_index

def get_helper_csv_nlp(data, column='text', text_path=None, target_path=None):
    
    helper = dict()
    
    helper[column] = data[column]
    # helper['cleaned_text'] = data['text'].apply(text_cleaning_with_numbers_as_text)
    helper['length'] = [len(i.split()) for i in data[column]]
    helper['readability'] = [textstat.flesch_kincaid_grade(i) for i in data[column]]
    
    if (column == "target") and (text_path != None):
        sim_scores = []
        gt_text = pd.read_csv(text_path)
        gt_target = pd.read_csv(target_path)
        for i in range(len(gt_text)):
            sim = cosine_similarity(X=[gt_text.iloc[i]], Y=[gt_target.iloc[i]])
            sim_scores.append(round(sim[0][0], 2))
        
        helper['similarity'] = sim_scores
        
    helper_df = pd.DataFrame(helper)
    return helper_df

def donut_for_spell_errors(value):
    df = {"Label": ["Spelling Error", "Correct"], "Value": [value, 100 - value]}
    df = pd.DataFrame(df)
    figure = px.pie(df, names='Label', values='Value', hole=0.8)
    return figure

def get_glcm_csv(path_df, distances, angles):
    
    """
    Takes in image directory paths as df then applies one distance to one angle not a list
    """
    contrast_list, energy_list, homogeneity_list, correlation_list, dissimilarity_list = [], [], [], [], []
    
    for i in path_df['paths']:
        image = cv.imread(i)
    
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, prop="contrast")
        energy = graycoprops(glcm, prop='energy')
        homogeneity = graycoprops(glcm, prop='homogeneity')
        correlation = graycoprops(glcm, prop='correlation')
        dissimilarity = graycoprops(glcm, prop='dissimilarity')
        
        contrast_list.append(contrast[0][0])
        energy_list.append(energy[0][0])
        homogeneity_list.append(homogeneity[0][0])
        correlation_list.append(correlation[0][0])
        dissimilarity_list.append(dissimilarity[0][0])
        
    path_df['contrast'] = contrast_list
    path_df['energy'] = energy_list
    path_df['homogeneity'] = homogeneity_list
    path_df['correlation'] = correlation_list
    path_df['dissimilarity'] = dissimilarity_list
    
    return path_df

def prep_and_predict(model,df,class_names,image_size=224):
    
    prediction = []
    probs = []
    for i in df['paths']:
        image = tf.io.read_file(i) # Reading the file
        image = tf.image.decode_image(image,channels=3) # Decode the read image to tensor and making sure to have 3 channels due to rgb
        image = tf.image.resize(image,size=(image_size,image_size))

        image = image/255. # Rescaling the data
        
        proba = model.predict(tf.expand_dims(image,axis=0))
        probs.append(round(proba[0][0],2)) # Expanding the dimensions to make sure it matches with the dimensions of the input data
        
        
        pred_class = class_names[int(tf.round(proba)[0][0])]
        
        prediction.append(pred_class)
    
    df['probs'] = probs
    df['target'] = prediction
    
    return df

def CV_data_quality(paths_df):
    
    resolutions, sharpness, brightness, size, noise, num_anomaly, is_anomaly = [], [], [], [] ,[], [], []
    
    for i in paths_df['paths']:
        
        image = cv.imread(i)
        
        resolution = (image.shape[1], image.shape[0])
        size_ = image.size
        
        sharp = cv.Laplacian(image, cv.CV_64F).var()
        bright = np.mean(image)
        noise_ = np.std(image)
        
        deviation = np.abs(image - np.mean(image)) / np.std(image)
        
        num_anomaly_ = np.sum(deviation > 3)
        is_anomaly_= num_anomaly_ > 2500
        
        resolutions.append(resolution)
        sharpness.append(sharp)
        brightness.append(bright)
        size.append(size_)
        noise.append(noise_)
        num_anomaly.append(num_anomaly_)
        is_anomaly.append(is_anomaly_)
        
    paths_df['resolution'] = resolutions
    paths_df['sharpness'] = sharpness
    paths_df['brightness'] = brightness
    paths_df['size'] = size
    paths_df['noise'] = noise
    paths_df['Number of Anomalies'] = num_anomaly
    paths_df['Anomaly'] = is_anomaly

    return paths_df

def CV_images_for_data_quality(paths_df):
    
    normal_path = paths_df[paths_df['Anomaly'] == False].sample(1)['paths'].iloc[0]
    anomaly_path = paths_df[paths_df['Anomaly'] == True].sample(1)['paths'].iloc[0]
    
    figures = dict()
    
    figures['max sharpness'] = cv.imread(paths_df.iloc[paths_df['sharpness'].argmax()]['paths'])
    figures['min sharpness'] = cv.imread(paths_df.iloc[paths_df['sharpness'].argmin()]['paths'])
    
    figures['max brightness'] = cv.imread(paths_df.iloc[paths_df['brightness'].argmax()]['paths'])
    figures['min brightness'] = cv.imread(paths_df.iloc[paths_df['brightness'].argmin()]['paths'])
    
    figures['max noise'] = cv.imread(paths_df.iloc[paths_df['noise'].argmax()]['paths'])
    figures['min noise'] = cv.imread(paths_df.iloc[paths_df['noise'].argmin()]['paths'])
    
    normal_image = cv.imread(normal_path)
    anomaly_image = cv.imread(anomaly_path)
    
    gray_normal = cv.cvtColor(normal_image, cv.COLOR_BGR2GRAY)
    pixels_normal = gray_normal.flatten()
    
    gray_anomaly = cv.cvtColor(anomaly_image, cv.COLOR_BGR2GRAY)
    pixels_anomaly = gray_anomaly.flatten()
    
    
    figure = go.Figure()
    figure.add_traces(go.Histogram(x=pixels_normal, nbinsx=256, name="Normal Image"))
    figure.add_traces(go.Histogram(x=pixels_anomaly, nbinsx=256, name="Anomaly Image"))
    
    figure.update_layout(barmode='overlay',
                        title="Histogram of Pixel Intensities",
                        xaxis_title="Pixel Intensity")
    
    figures['Histogram'] = figure
    
    return figures

def get_text_scores_df(gt, prod, gt_embed, prod_embed, gt_input):

  bleu_scores, rouge_r, rouge_p, rouge_f, sim_scores, sim_text_scores = [], [], [], [], [], []
  
  rouge = Rouge()

  for i in range(len(gt)):

    bleu_scores.append(round(sentence_bleu(gt['target'].iloc[i].split(), prod['target'].iloc[i].split()), 4))

    rouge_score = rouge.get_scores(gt['target'].iloc[i], prod['target'].iloc[i])[0]

    rouge_r.append(round(rouge_score['rouge-1']['r'], 2))
    rouge_p.append(round(rouge_score['rouge-1']['p'], 2))
    rouge_f.append(round(rouge_score['rouge-1']['f'], 2))

    sim = cosine_similarity(X=[gt_embed.iloc[i]], Y=[prod_embed.iloc[i]])
    sim_scores.append(round(sim[0][0], 2))
    
    another_sim = cosine_similarity(X=[gt_input.iloc[i]], Y=[prod_embed.iloc[i]])
    sim_text_scores.append(round(another_sim[0][0], 2))

  text_scores_df = pd.DataFrame({"bleu": bleu_scores,
                                "rouge_recall": rouge_r,
                                 "rouge_precision": rouge_p,
                                 "rouge_F1": rouge_f,
                                "similarity_with_gt": sim_scores,
                                "similarity_wth_text": sim_text_scores})
  return text_scores_df

def text_score_figures(scores_df):

  figures = dict()

  figure_hist_rouge = go.Figure()

  figure_hist_rouge.add_traces(go.Histogram(x=scores_df['rouge_recall'], nbinsx=10, name="Recall"))
  figure_hist_rouge.add_traces(go.Histogram(x=scores_df['rouge_precision'], nbinsx=10, name="Precision"))
  figure_hist_rouge.add_traces(go.Histogram(x=scores_df['rouge_F1'], nbinsx=10, name="F1"))

  figure_hist_rouge.update_layout(barmode='overlay',
                        title="ROUGE Scores",
                        xaxis_title="Score")

  figure_hist_bleu = go.Figure()
  figure_hist_bleu.add_traces(go.Histogram(x=scores_df['bleu'], nbinsx=10, name="BLEU Score"))

  figure_hist_similarity = go.Figure()
  figure_hist_similarity.add_traces(go.Histogram(x=scores_df['similarity_with_gt'], nbinsx=10, name="Similarity with Ground Truth"))
  
  figure_hist_similarity_t = go.Figure()
  figure_hist_similarity_t.add_traces(go.Histogram(x=scores_df['similarity_wth_text'], nbinsx=10, name="Similarity with Text"))

  figures['ROUGE Box'] = px.box(scores_df,['rouge_F1', 'rouge_recall', 'rouge_precision'], points="all")
  figures['ROUGE Hist'] = figure_hist_rouge

  figures['BLEU Box'] = px.box(scores_df, 'bleu', points="all")
  figures['BLEU Hist'] = figure_hist_bleu

  figures['Similarity Box'] = px.box(scores_df, 'similarity_with_gt', points="all")
  figures['Similarity Hist'] = figure_hist_similarity
  
  figures['Similarity_t Box'] = px.box(scores_df, 'similarity_wth_text', points="all")
  figures['Similarity_t Hist'] = figure_hist_similarity_t

  return figures

def seq2seq_metrics(gt_target, prod_target):

  metrics = dict()
  rouge = Rouge()
  scores = rouge.get_scores(gt_target, prod_target, avg=True)

  metrics['Precision'] = round(scores['rouge-1']['p'] * 100, 2)
  metrics['F1'] = round(scores['rouge-1']['f'] * 100, 2)
  metrics['Recall'] = round(scores['rouge-1']['r'] * 100, 2)

  metrics['BLEU'] = round(corpus_bleu(gt_target, prod_target), 2)

  return metrics