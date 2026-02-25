import os
import re
import ast
import json
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib as mpl

from openai import OpenAI
from tqdm import tqdm
from util import load_geojson, get_geo_id
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix

county_geojson = '../data/resources/county.geojson'
state_geojson = '../data/resources/state-us.geojson'
country_geojson = '../data/resources/countries.geojson'

county_polygons = load_geojson(county_geojson, county_geojson, state_geojson)
state_polygons = load_geojson(state_geojson, county_geojson, state_geojson)
country_polygons = load_geojson(country_geojson, county_geojson, state_geojson)


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
            geoid = feature.get('id')
        geoids.add(geoid)
    return geoids

def get_all_geo_labels(county_geojson, state_geojson, country_geojson):
    county_ids = load_geoids(county_geojson, county_geojson, state_geojson)
    state_ids = load_geoids(state_geojson, county_geojson, state_geojson)
    country_ids = load_geoids(country_geojson, county_geojson, state_geojson)

    all_geo_labels = county_ids.union(state_ids).union(country_ids)
    print("Total labels:", len(all_geo_labels))
    return all_geo_labels

def load_dataset(csv_path):
    logging.info(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    logging.info(f"Dataset loaded with {len(df)} rows.")
    return df


def extract_city_state_from_location(df, location_col= "location"):
    logging.info("Extracting city and state from location column...")

    def extract(loc_str):
        try:
            loc = ast.literal_eval(loc_str)
            return pd.Series([loc.get('city', ''), loc.get('state', '')])
        except Exception as e:
            logging.error(f"Failed to parse location: {loc_str} -> {e}")
            return pd.Series(['', ''])

    df[['publisher_city', 'publisher_state']] = df[location_col].apply(extract)
    return df


def generate_prompt(title, content, city, state):
    return f"""
            The **geo-focus** refers to the **main region(s) most directly associated with the subject in the article**, regardless of where it was published.

            The following article is from a US local news publisher located in {city}, {state}.

            Return the coordinates (in decimal format) of all geo-focused locations along with the administrative level (county, state, or country). Then indicate the geo-focus level (local, state, national, international, or none) w.r.t to U.S. context.

            ---

            **local**:
            - Articles centered on a specific city, town, or county.

            **state**:
            - Articles focused on **multiple locations within the publisher’s U.S. state**, or on **statewide issues** that affect regions across that state.

            **national**:
            - Articles involving **multiple U.S. states**, **nationwide issues**, or **events primarily concerning states other than the publisher’s**.

            **international**:
            - Articles primarily concerning **events, issues, or entities outside the United States**.

            **none**:
            - Articles with **no meaningful or identifiable geographic focus**.

            ---

            **Important:**
            - Base your classification only on what the article is **about**, not where it was published.
            - Pay close attention to whether content is about the **publisher’s state** (state) or **other states** (national).
            - If multiple locations exist at the same level, return **all coordinates**, one per line.
            - Only return locations if applicable; if not, return only: geo_focus_level:none

            ---

            Now, classify the article:

            **Title:** {title}  
            **Content:** {content}  
            **Publisher Location:** {city}, {state}

            ---

            **Your answer:**  
            Return coordinates and geo-focus level.  
            Format:
            latitude: <value>, longitude: <value>, type: <county/state/country>
            latitude: <value>, longitude: <value>, type: <county/state/country>
            geo_focus_level:<local/state/national/international/none>
            """


def geo_focus_with_gpt(title, content, city, state, 
                       county_polygons, state_polygons, country_polygons):

    print("Geo-focus Identification...")
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=key)
        prompt = generate_prompt(title, content, city, state)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=600  
        )
        response_text = response.choices[0].message.content.strip()

        print(response_text)

        pattern = r"latitude\s*:\s*([-+]?\d*\.\d+|\d+)\s*,\s*longitude\s*:\s*([-+]?\d*\.\d+|\d+)\s*,\s*type\s*:\s*(county|state|country)"
        matches = re.findall(pattern, response_text, re.IGNORECASE)

        if not matches:
            print("Could not parse GPT response.")
            return None

        geo_location_data = []
        for match in matches:
            latitude = float(match[0])
            longitude = float(match[1])
            admin_type = match[2].lower()
            geo_location_data.append({
                "latitude": latitude,
                "longitude": longitude,
                "type": admin_type
            })

        focus_pattern = r"geo_focus_level\s*:\s*(local|state|national|international|none)"
        focus_match = re.search(focus_pattern, response_text, re.IGNORECASE)
        geo_focus_level = focus_match.group(1).lower() if focus_match else "none"

        geo_locations = []
        if geo_focus_level == "none":
            geo_locations = ['none']
        else:
            for geo_location in geo_location_data:
                latitude = geo_location["latitude"]
                longitude = geo_location["longitude"]
                admin_type = geo_location["type"]

                geo_id = None  

                if geo_focus_level == "none":
                    geo_id = 'none'
                elif geo_focus_level == "local" and admin_type == "county":
                    geo_id = get_geo_id(longitude, latitude, county_polygons)

                elif geo_focus_level == "state":
                    if admin_type == "state":
                        geo_id = get_geo_id(longitude, latitude, state_polygons)
                    elif admin_type == "county":
                        geo_id = get_geo_id(longitude, latitude, county_polygons)

                elif geo_focus_level == "national":
                    if admin_type == "state":
                        geo_id = get_geo_id(longitude, latitude, state_polygons)
                    elif admin_type == "county":
                        geo_id = get_geo_id(longitude, latitude, county_polygons)
                    elif admin_type == "country":
                        geo_id = get_geo_id(longitude, latitude, country_polygons)
                        if geo_id != "USA":
                            geo_id = None  

                elif geo_focus_level == "international" and admin_type == "country":
                    geo_id = get_geo_id(longitude, latitude, country_polygons)
                    if geo_id == "USA":
                        geo_id = None  

                if geo_id:
                    geo_locations.append(geo_id)

        return {
            "geo_location_data": geo_location_data,
            "geo_focus_level": geo_focus_level,
            "geo_locations": geo_locations
        }

    except Exception as e:
        print(f"Error identifying geo-focus: {e}")
        return None
    

def classify_articles(df):
    logging.info("Starting classification...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title = row.get("title", "")
        content = row.get("content", "")
        city = row.get("publisher_city", "")
        state = row.get("publisher_state", "")

        geo_focus_data = geo_focus_with_gpt(
            title, content, city, state,
            county_polygons, state_polygons, country_polygons)

        if geo_focus_data:  
            gpt_geo_location_data = geo_focus_data.get("geo_location_data", [])
            gpt_geo_focus_level = geo_focus_data.get("geo_focus_level", "none")
            gpt_geo_locations = geo_focus_data.get("geo_locations", ['none'])
        else:
            gpt_geo_location_data = []
            gpt_geo_focus_level = "none"
            gpt_geo_locations = ['none']

        df.at[idx, "gpt_geo_location_data"] = json.dumps(gpt_geo_location_data)
        df.at[idx, "gpt_geo_focus_level"] = gpt_geo_focus_level
        df.at[idx, "gpt_geo_locations"] = json.dumps(gpt_geo_locations)

    return df


def save_results(df, output_path):
    df.to_csv(output_path, index=False)
    logging.info(f"\nClassification complete. Results saved to: {output_path}")


def analyze_gpt_geo_focus_level_predictions(csv_path):

    df = pd.read_csv(csv_path)
    logging.info("Columns in the DataFrame:", df.columns.tolist())

    df["gpt_geo_focus_level"] = df["gpt_geo_focus_level"].str.strip().str.lower()
    df["label"] = df["label"].str.strip().str.lower()

    df = df.dropna(subset=["label", "gpt_geo_focus_level"])
    df = df[df["gpt_geo_focus_level"] != "error"]

    labels = ["international", "national", "state", "local", "none"]
    display_labels = ["intl.", "national", "state", "local", "none"]
    min_count = df["label"].value_counts().min()
    balanced_df = df.groupby("label").sample(n=min_count, random_state=42)

    y_true = balanced_df["label"]
    y_pred = balanced_df["gpt_geo_focus_level"]

    print(f"Balanced dataset size per class: {min_count}")
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report (ordered):")
    print(classification_report(y_true, y_pred, labels=labels, digits=2))

    font_files = fm.findSystemFonts(fontpaths=["./fonts/"])
    for font_file in font_files:
        if "helvetica" in font_file.lower():
            fm.fontManager.addfont(font_file)
            print("Loaded font:", font_file)

    mpl.rcParams['font.family'] = 'Helvetica'

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"fontsize": 48}   
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28) 

    ax.set_xlabel('Predicted Label', fontsize=28)
    ax.set_ylabel('True Label', fontsize=28)

    plt.xticks(fontsize=28)
    plt.yticks(rotation=90, fontsize=28)

    plt.tight_layout()
    plt.savefig("../results/gpt/con_matrix_gpt.png")
    plt.show()

def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def evaluate(df, mlb):
    df['geo_focus_label'] = df['geo_focus_label'].apply(parse_list)
    df['gpt_geo_locations'] = df['gpt_geo_locations'].apply(parse_list)

    y_true = df['geo_focus_label'].tolist()
    y_pred = df['gpt_geo_locations'].tolist()

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    precision = precision_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    return precision, recall, f1

def analyse_geo_focus_prediction(df):
    all_geo_labels = set(get_all_geo_labels(county_geojson, state_geojson, country_geojson))
    all_geo_labels.add('none')

    mlb = MultiLabelBinarizer(classes=list(all_geo_labels))
    mlb.fit(list(all_geo_labels))

    precision, recall, f1 = evaluate(df, mlb)
    print("\nFinal Test Set Evaluation")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

def fix_empty(x):
    try:
        lst = ast.literal_eval(x) if isinstance(x, str) else x
        if not lst:
            return ["none"]
        return lst
    except:
        return ["none"]

def gpt_geo_focus_classification(input_csv):
    df = load_dataset(input_csv)
    df = extract_city_state_from_location(df)
    # df = classify_articles(df)
    # save_results(df, output_csv)
    analyze_gpt_geo_focus_level_predictions(input_csv)
    analyse_geo_focus_prediction(df)

gpt_geo_focus_classification("../results/gpt/gpt-data.csv")

