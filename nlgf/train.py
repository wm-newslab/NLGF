import os
import ast
import shap
import joblib
import json
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from util import evaluate_geo_focus, extract_features
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import matplotlib as mpl
import matplotlib.font_manager as fm

# -----------------------------
# Register Helvetica fonts
# -----------------------------
font_files = fm.findSystemFonts(fontpaths=["./fonts/"])
for font_file in font_files:
    if "helvetica" in font_file.lower():
        fm.fontManager.addfont(font_file)

# Use Helvetica with sensible fallbacks
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# -----------------------------
# Increase global font sizes
# -----------------------------
mpl.rcParams.update({
    "font.size": 16,          # base font
    "axes.titlesize": 16,     # plot title
    "axes.labelsize": 16,     # x & y labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})



county_geojson = '../data/resources/county.geojson'
state_geojson = '../data/resources/state-us.geojson'
country_geojson = '../data/resources/countries.geojson'

data_file = "../data/data.csv"
model_path = "../results/model"

feature_cols = [
    "title_topo_cnt_intl", "title_topo_cnt_national", "title_topo_cnt_state", "title_topo_cnt_local",
    "intl_igl_cnt", "national_igl_cnt", "state_igl_cnt", "local_igl_cnt",
    "leading_topo_intl_igl_cnt", "leading_topo_national_igl_cnt", "leading_topo_state_igl_cnt", "leading_topo_local_igl_cnt",
    "uniq_intl_igl", "uniq_national_igl", "uniq_local_igl"
]


def safe_parse_toponym_scores(x):
    if isinstance(x, (dict, list)):
        return x
    if x is None:
        return {}
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            import json
            return json.loads(s)
        except Exception:
            return s


def load_train_data(filepath, model_path):
    print(f"Loading data from: {filepath}")

    df = pd.read_csv(filepath)

    df['label'] = df['label'].astype(str).str.lower()

    print(df['label'].value_counts())

    df['toponym_scores'] = df.get('toponym_scores', pd.Series([{}] * len(df)))
    df['toponym_scores'] = df['toponym_scores'].apply(safe_parse_toponym_scores)

    class_to_index = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['y_encoded'] = df['label'].map(class_to_index)

    features = df['toponym_scores'].apply(extract_features)
    input_data = pd.DataFrame(list(features))

    for col in input_data.columns:
        df[col] = input_data[col]

    X = df[feature_cols]
    y = df['y_encoded']

    return df, X, y, class_to_index



def balance_data(X, y_encoded):
    print("Balancing dataset via undersampling...")
    data = pd.concat([X.reset_index(drop=True), pd.Series(y_encoded, name='label').reset_index(drop=True)], axis=1)
    min_class_size = int(data['label'].value_counts().min())
    balanced_data = (
        data.groupby('label', group_keys=False)
            .apply(lambda g: g.sample(n=min_class_size, random_state=42))
            .reset_index(drop=True)
    )
    balanced_data = balanced_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_bal = balanced_data[X.columns]
    y_bal = balanced_data['label'].astype(int)
    print(f"Balanced shape: {X_bal.shape} | Label distribution: {np.bincount(y_bal)}\n")
    return X_bal, y_bal


def tune_hyperparameters(X, y):
    print("\nTuning hyperparameters...")
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
    print("Evaluating final model with cross validation...")
    all_y_true, all_y_pred = [], []
    save_path = os.path.join(model_path, "conf_matrix.png")

    LABEL_ORDER = ["international", "national", "state", "local", "none"]
    DISPLAY_LABELS = ["intl.", "national", "state", "local", "none"]

    inv_index_to_class = {int(k): v for k, v in index_to_class.items()}
    class_to_index = {v: k for k, v in inv_index_to_class.items()}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"Train accuracy: {clf.score(X_train, y_train):.3f}")
        print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")

    all_y_true_labels = [inv_index_to_class[i] for i in all_y_true]
    all_y_pred_labels = [inv_index_to_class[i] for i in all_y_pred]

    print("\nClassification Report:\n")
    print(
        classification_report(
            all_y_true_labels,
            all_y_pred_labels,
            labels=LABEL_ORDER,
            target_names=DISPLAY_LABELS,
            digits=2,
        )
    )

    font_files = fm.findSystemFonts(fontpaths=["./fonts/"])
    for font_file in font_files:
        if "helvetica" in font_file.lower():
            fm.fontManager.addfont(font_file)
    mpl.rcParams["font.family"] = "Helvetica"

    ordered_numeric = [class_to_index[label] for label in LABEL_ORDER]

    cm = confusion_matrix(
        all_y_true,
        all_y_pred,
        labels=ordered_numeric,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=DISPLAY_LABELS,
        yticklabels=DISPLAY_LABELS,
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 48},
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)

    ax.set_xlabel("Predicted Label", fontsize=28)
    ax.set_ylabel("True Label", fontsize=28)

    plt.xticks(fontsize=28)
    plt.yticks(rotation=90, fontsize=28)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Confusion matrix saved to: {save_path}")



def save_geo_focus_level_results(df, X, final_model, index_to_class, model_path):
    print("Generating geo focus level predictions for each data point")
    try:
        getattr(final_model, "predict")
    except Exception:
        raise RuntimeError("Model does not appear to be fitted yet.")

    predictions = final_model.predict(X)
    inv_index_to_class = {int(k): v for k, v in index_to_class.items()}
    df['predicted_label'] = [inv_index_to_class.get(int(p), str(p)) for p in predictions]

    return df

def load_geoids(geojson_file, county_geojson, state_geojson):
    with open(geojson_file, encoding="utf-8") as f:
        geojson_data = json.load(f)

    geoids = set()
    for feature in geojson_data['features']:
        if geojson_file == county_geojson:
            geoid = feature['properties']['GEOID']
        elif geojson_file == state_geojson:
            geoid = feature['properties']['NAME']
        else:  
            geoid = feature.get('id', feature['properties'].get('ISO_A3'))
        geoids.add(geoid)
    return geoids

def get_all_geo_labels(county_geojson, state_geojson, country_geojson):
    county_ids = load_geoids(county_geojson, county_geojson, state_geojson)
    state_ids = load_geoids(state_geojson, county_geojson, state_geojson)
    country_ids = load_geoids(country_geojson, county_geojson, state_geojson)

    all_geo_labels = county_ids.union(state_ids).union(country_ids)
    print("Total labels:", len(all_geo_labels))
    return all_geo_labels

def fix_empty(x):
    try:
        lst = ast.literal_eval(x) if isinstance(x, str) else x
        if not lst:
            return ["none"]
        return lst
    except:
        return ["none"]

def save_shap_feature_importance(model, X, model_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)
    else:
        shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        shap_array = np.mean(np.abs(shap_array), axis=-1)
    elif shap_array.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_array.shape}")

    mean_abs_shap = shap_array.mean(axis=0)

    feature_names = model.get_booster().feature_names

    if len(mean_abs_shap) != len(feature_names):
        raise ValueError(
            f"Feature mismatch: SHAP={len(mean_abs_shap)} vs model={len(feature_names)}"
        )

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_csv = os.path.join(model_path, "feature_importance_shap.csv")
    shap_df.to_csv(shap_csv, index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=shap_df.head(15),
        x="mean_abs_shap",
        y="feature",
        color="#1f77b4"
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("")

    plt.tight_layout()

    shap_png = os.path.join(model_path, "feature_importance_shap.png")
    plt.savefig(shap_png, dpi=300)
    plt.show()

    print(f"SHAP feature importance saved to:\n- {shap_csv}\n- {shap_png}")


def print_shap_feature_importance(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)
    else:
        shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        shap_array = np.mean(np.abs(shap_array), axis=-1)
    elif shap_array.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_array.shape}")

    mean_abs_shap = shap_array.mean(axis=0)
    feature_names = model.get_booster().feature_names

    pairs = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nSHAP Feature Importance (Mean |SHAP|):")
    for feat, val in pairs:
        print(f"{feat:35s} {val:.6f}")


if __name__ == "__main__":
    df, X, y, class_to_index = load_train_data(data_file, model_path)

    X_bal, y_bal = balance_data(X, y)

    best_params, skf = tune_hyperparameters(X_bal, y_bal)
    final_model = XGBClassifier(eval_metric='mlogloss', random_state=42, **best_params)

    final_model.fit(X_bal, y_bal)

    save_shap_feature_importance(
        final_model,
        X_bal,
        model_path
    )

    print_shap_feature_importance(final_model, X_bal)

    index_to_class = {v: k for k, v in class_to_index.items()}
    class_names = [index_to_class[i] for i in sorted(index_to_class.keys())]
    
    evaluate_model(X_bal, y_bal, final_model, skf, index_to_class, list(class_to_index.keys()), model_path)

    df = save_geo_focus_level_results(df, X, final_model, index_to_class, model_path)

    df_all = evaluate_geo_focus(df)

    output_file = os.path.join(model_path, 'gf_data_with_prediction.csv')
    df_all.to_csv(output_file, index=False)
    print(f"Predictions with labels saved to {output_file}")

    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, 'labels.json'), 'w') as f:
        json.dump(class_to_index, f)

    joblib.dump(final_model, os.path.join(model_path, 'nlfg.pkl'))
    print(f"Model, encoder, and top features saved to the path: {model_path}")

