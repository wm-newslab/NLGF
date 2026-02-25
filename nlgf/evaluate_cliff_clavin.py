import ast
import json
import pandas as pd
import urllib.parse
import requests

from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_score, recall_score, f1_score
)

from util import get_all_geo_labels


def safe_literal_eval(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            try:
                return json.loads(v)
            except Exception:
                return v
    return v


def clean_and_encode_text(text: str) -> str:
    cleaned = (
        str(text)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .strip()
    )
    encoded = urllib.parse.quote(cleaned, safe="/:@&?=+,$-_.!~*'()")
    return encoded


def load_geojson(geojson_file, county_geojson, state_geojson):
    with open(geojson_file, encoding="utf-8") as f:
        geojson_data = json.load(f)

    polygons = {}
    for feature in geojson_data["features"]:
        geometry_type = feature["geometry"]["type"]
        coordinates = feature["geometry"]["coordinates"]

        if geojson_file == county_geojson:
            geoid = feature["properties"]["GEOID"]
        elif geojson_file == state_geojson:
            geoid = feature["properties"]["NAME"]
        else:
            geoid = feature["id"]

        if geometry_type == "Polygon":
            polygons[geoid] = Polygon(coordinates[0])
        elif geometry_type == "MultiPolygon":
            polygons[geoid] = MultiPolygon([Polygon(coords[0]) for coords in coordinates])

    return polygons


def get_geo_id(long, lat, polygons):
    try:
        point = Point(float(long), float(lat))
    except ValueError:
        return None

    for geo_id, polygon in polygons.items():
        try:
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            if polygon.contains(point):
                return geo_id
        except Exception:
            continue
    return None


def run_cliff_clavin(df, county_polygons, state_polygons, country_polygons, enable=False):
    if not enable:
        return df

    geo_focus_list = []
    geo_cities = []
    geo_states = []
    geo_countries = []

    for i, article_text in tqdm(enumerate(df["content"]), total=len(df)):
        try:
            encoded_text = clean_and_encode_text(article_text)

            url = "http://localhost:8080/cliff-2.6.1/parse/text"
            data = {"q": article_text}

            parse_response = requests.post(url, data=data)

            if parse_response.status_code == 200:
                parse_result = parse_response.json()
                geo_focus = parse_result.get("results", {}).get("places", {}).get("focus", {})
            else:
                print(f"[Row {i}] Error {parse_response.status_code}")
                geo_focus = {}
        except Exception as e:
            print(f"[Row {i}] Exception: {e}")
            geo_focus = {}

        geo_focus_list.append(geo_focus)

    df["geo_focus_cc_v2"] = geo_focus_list

    for i, row in tqdm(df.iterrows(), total=len(df)):
        focus = row.get("geo_focus_cc_v2")

        try:
            focus_dict = eval(focus) if isinstance(focus, str) else focus
        except Exception:
            focus_dict = {}

        city_geoids = set()
        state_geoids = set()
        country_geoids = set()

        for city in focus_dict.get("cities", []):
            lat, lon = city.get("lat"), city.get("lon")
            cc = city.get("countryCode")
            if cc == "US":
                geoid = get_geo_id(lon, lat, county_polygons)
                if geoid:
                    city_geoids.add(geoid)
            else:
                geoid = get_geo_id(lon, lat, country_polygons)
                if geoid:
                    country_geoids.add(geoid)

        for state in focus_dict.get("states", []):
            lat, lon = state.get("lat"), state.get("lon")
            cc = state.get("countryCode")
            if cc == "US":
                geoid = get_geo_id(lon, lat, state_polygons)
                if geoid:
                    state_geoids.add(geoid)
            else:
                geoid = get_geo_id(lon, lat, country_polygons)
                if geoid:
                    country_geoids.add(geoid)

        for country in focus_dict.get("countries", []):
            lat, lon = country.get("lat"), country.get("lon")
            cc = country.get("countryCode")
            if cc == "US":
                country_geoids.add("USA")
            else:
                geoid = get_geo_id(lon, lat, country_polygons)
                if geoid:
                    country_geoids.add(geoid)

        geo_cities.append(sorted(list(city_geoids)))
        geo_states.append(sorted(list(state_geoids)))
        geo_countries.append(sorted(list(country_geoids)))

    df["geo_cities"] = geo_cities
    df["geo_states"] = geo_states
    df["geo_countries"] = geo_countries

    return df


def merge_geo_focus_columns(df):
    for col in ["geo_cities", "geo_states", "geo_countries"]:
        df[col] = df[col].apply(safe_literal_eval)

    df["geo_focus_prediction_cc"] = df.apply(
        lambda r: (r["geo_cities"] or [])
                + (r["geo_states"] or [])
                + (r["geo_countries"] or []),
        axis=1
    )
    return df


def evaluate_predictions(df, all_geo_labels):
    df["geo_focus_label"] = df["geo_focus_label"].apply(safe_literal_eval)
    df["geo_focus_prediction_cc"] = df["geo_focus_prediction_cc"].apply(safe_literal_eval)

    mlb = MultiLabelBinarizer(classes=list(all_geo_labels))
    mlb.fit(list(all_geo_labels))

    y_true = df["geo_focus_label"].tolist()
    y_true_bin = mlb.transform(y_true)

    y_pred = df["geo_focus_prediction_cc"].tolist()
    y_pred_bin = mlb.transform(y_pred)

    precision = precision_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)

    print("Final Test Set Evaluation")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

if __name__ == "__main__":

    county_geojson = "../data/resources/county.geojson"
    state_geojson = "../data/resources/state-us.geojson"
    country_geojson = "../data/resources/countries.geojson"

    print("Loading GeoJSON boundaries...")
    county_polygons = load_geojson(county_geojson, county_geojson, state_geojson)
    state_polygons = load_geojson(state_geojson, county_geojson, state_geojson)
    country_polygons = load_geojson(country_geojson, county_geojson, state_geojson)
    print("GeoJSONs loaded.")

    INPUT_CSV = "../results/cc/cc-data.csv"
    df = pd.read_csv(INPUT_CSV)

    df = run_cliff_clavin(
        df,
        county_polygons,
        state_polygons,
        country_polygons,
        enable=False   
    )

    df = merge_geo_focus_columns(df)

    all_geo_labels = get_all_geo_labels(county_geojson, state_geojson, country_geojson)
    all_geo_labels.add("none")

    evaluate_predictions(df, all_geo_labels)

    print("Processing complete.")
