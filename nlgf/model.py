import os
import shap
import joblib
import json
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from util import get_features, get_geo_focus_label, evaluate_geo_focus, get_county_name, get_country_name

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

feature_cols = [
    "title_topo_cnt_intl", "title_topo_cnt_national", "title_topo_cnt_state", "title_topo_cnt_local",
    "intl_igl_cnt", "national_igl_cnt", "state_igl_cnt", "local_igl_cnt",
    "leading_topo_intl_igl_cnt", "leading_topo_national_igl_cnt", "leading_topo_state_igl_cnt", "leading_topo_local_igl_cnt",
    "uniq_intl_igl", "uniq_national_igl", "uniq_local_igl"
]

def load_train_data(filepath):
    print(f"Loading data from: {filepath}")

    df = pd.read_csv(filepath)

    df['label'] = df['label'].astype(str).str.lower()  
    df['label'] = df['label'].replace(['nan', '', ' '], 'none')  

    df = df[df['label'] != "none"]

    print("Number of entries per label (after removing 'none'):")
    print(df['label'].value_counts())

    class_to_index = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['y_encoded'] = df['label'].map(class_to_index)

    X = df[feature_cols]  
    y = df['y_encoded']

    return df, X, y, class_to_index


def analyze_feature_correlations(X, y, correlation_threshold=0.9):
    print("Computing full feature correlations...")
    correlation_matrix = X.corr()

    print("Computing absolute correlations for feature reduction...")
    abs_corr_matrix = correlation_matrix.abs()
    upper_triangle = abs_corr_matrix.where(np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > correlation_threshold)]
    
    print(f"Highly correlated features to drop (threshold > {correlation_threshold}): {to_drop}")
    X = X.drop(columns=to_drop)
    return X, y


def balance_data(X, y_encoded):
    print("Balancing dataset via undersampling...")
    data = pd.concat([X, pd.Series(y_encoded, name='label')], axis=1)
    min_class_size = data['label'].value_counts().min()
    balanced_data = data.groupby('label').sample(n=min_class_size, random_state=42)
    X_bal = balanced_data[X.columns]
    y_bal = balanced_data['label']
    print(f"Balanced shape: {X_bal.shape} | Label distribution: {np.bincount(y_bal)}")
    return X_bal, y_bal


def get_top_features(X, y, importance_threshold=0.005):
    print("Calculating permutation feature importance...")
    model = XGBClassifier(eval_metric='mlogloss', random_state=42, max_depth=3)
    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='f1_macro')
    
    importances_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm.importances_mean
    }).sort_values(by='importance', ascending=False)

    top_features_df = importances_df[importances_df['importance'] > importance_threshold]
    print(f"Selected {len(top_features_df)} features with importance > {importance_threshold}:")
    print(f"\n{top_features_df}")

    return top_features_df['feature'].tolist()


def tune_hyperparameters(X, y):
    print("Tuning hyperparameters...")
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10],
        'n_estimators': [25, 50, 100, 200],
        'subsample': [0.8, 0.9, 1.0]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        XGBClassifier(eval_metric='mlogloss', random_state=42),
        param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_params_, skf


def evaluate_model(X, y, model, skf, index_to_class, class_names, model_path):
    
    print("Evaluating final model with cross-validation...")
    all_y_true, all_y_pred = [], []
    save_path = f'{model_path}/conf_matrix.png'
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
        print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

    class_names = [c if c != "international" else "intl." for c in class_names]
    index_to_class = {i: (c if c != "international" else "intl.") for i, c in index_to_class.items()}

    all_y_true_labels = [index_to_class[i] for i in all_y_true]
    all_y_pred_labels = [index_to_class[i] for i in all_y_pred]


    print("Classification Report:")
    print("\n" + classification_report(
        all_y_true_labels,
        all_y_pred_labels,
        target_names=class_names,
        digits=3
    ))

    cm_to_plot = confusion_matrix(all_y_true, all_y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"fontsize": 20}
    )
    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(rotation=90, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Confusion matrix saved to: {save_path}")


def plot_shap_summary(X, y, final_model, top_features, label_encoder, output_path):
    
    logging.debug("Fitting model and computing SHAP values...")
    final_model.fit(X, y)
    
    explainer = shap.Explainer(final_model, X)
    shap_values = explainer(X)

    logging.debug("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X, feature_names=top_features, show=False)

    plt.yticks(fontsize=10)
    plt.xlabel("Average Impact of Feature on Prediction", fontsize=10)
    plt.xticks(fontsize=10)

    class_names = label_encoder.classes_
    legend = plt.gca().get_legend()
    if legend:
        for i, text in enumerate(legend.get_texts()):
            text.set_text(class_names[i])
            text.set_fontsize(10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def save_geo_focus_level_results(df, X, final_model, index_to_class, model_path):
    
    print("Generating geo focus levelpredictions for each data point")
    predictions = final_model.predict(X)

    df['predicted_label'] = [index_to_class[p] for p in predictions]

    output_file = os.path.join(model_path, 'gf_data_with_prediction.csv')
    df.to_csv(output_file, index=False)

    print(f"Predictions with labels saved to {output_file}")
    return df

def train(data_file, model_path):
    df, X, y, class_to_index = load_train_data(data_file)
    X_bal, y_bal = balance_data(X, y)

    best_params, skf = tune_hyperparameters(X_bal, y_bal)
    final_model = XGBClassifier(eval_metric='mlogloss', random_state=42, **best_params)

    index_to_class = {idx: label for label, idx in class_to_index.items()}
    evaluate_model(X_bal, y_bal, final_model, skf, index_to_class, list(class_to_index.keys()), model_path)
    df = save_geo_focus_level_results(df, X, final_model, index_to_class, model_path)

    evaluate_geo_focus(df)

    os.makedirs(model_path, exist_ok=True)
    
    with open(os.path.join(model_path, 'labels.json'), 'w') as f:
        json.dump(class_to_index, f)
    joblib.dump(final_model, os.path.join(model_path, 'nlfg.pkl'))
    print(f"Model, encoder, and top features saved to disk.")


def predict(link, publisher_longitude, publisher_latitude, model_path="../results/model" ):
    nlgf_path = os.path.join(model_path, 'nlfg.pkl')
    if not os.path.exists(nlgf_path):
        raise FileNotFoundError(f"Model file not found: {nlgf_path}")

    model = joblib.load(nlgf_path)

    with open(os.path.join(model_path, 'labels.json'), 'r') as f:
        class_to_index = json.load(f)

    features, toponym_scores = get_features(link, publisher_longitude, publisher_latitude)
    if isinstance(features, dict):
        features = [features]  
    input_data = pd.DataFrame(features)

    X_new = input_data[feature_cols]
    predictions = model.predict(X_new)

    index_to_class = {idx: label for label, idx in class_to_index.items()}
    predicted_label = [index_to_class[idx] for idx in predictions]

    geo_ids = get_geo_focus_label(toponym_scores, predicted_label[0])

    if predicted_label[0] == "local":
        geo_ids = [get_county_name(gid) for gid in geo_ids]
    elif predicted_label[0] == "international":
        geo_ids = [get_country_name(gid) for gid in geo_ids]

    return predicted_label[0], geo_ids