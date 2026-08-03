import os
import re
import ast
import gzip
import json
import spacy
import uuid
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from openai import OpenAI
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
from storysniffer import StorySniffer
from newspaper import Article
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

DEFAULT_HUGGINGFACE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HUGGINGFACE_MODEL_ALIASES = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}
_hf_client = None
_hf_client_token = None

county_geojson = '../data/resources/county.geojson'
state_geojson = '../data/resources/state-us.geojson'
country_geojson = '../data/resources/countries.geojson'

def parallel_ner(nlp, title, content):
    """
    Performs Named Entity Recognition (NER) on both the title and content.
    """

    ent_labels = ['FAC', 'GPE', 'LOC']    
    title_doc = nlp(title)
    title_entities = []
    logging.debug(f"title: {title}")
    for ent in title_doc.ents:
        if ent.label_ in ent_labels:
            title_entities.append({
                "entity": ent.text,
                "class": ent.label_,
                "is_from_title": True,
                "context": {
                    "sents": [{"sent": sent.text} for sent in ent.sent.as_doc().sents]
                }
            })

    content_doc = nlp(content)
    content_entities = []
    for ent in content_doc.ents:
        if ent.label_ in ent_labels:
            content_entities.append({
                "entity": ent.text,
                "class": ent.label_,
                "is_from_title": False,
                "context": {
                    "sents": [{"sent": sent.text} for sent in ent.sent.as_doc().sents]
                }
            })

    return title_entities + content_entities

def load_geojson(geojson_file, county_geojson, state_geojson):
    with open(geojson_file, encoding="utf-8") as f:
        geojson_data = json.load(f)
    
    polygons = {}
    for feature in geojson_data['features']:
        geometry_type = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']

        if geojson_file == county_geojson:
            geoid = feature['properties']['GEOID']
        elif geojson_file == state_geojson:
            geoid = feature['properties']['NAME']
        else:
            geoid = feature['id']
        
        if geometry_type == 'Polygon':
            polygons[geoid] = Polygon(coordinates[0])
        elif geometry_type == 'MultiPolygon':
            polygons[geoid] = MultiPolygon([Polygon(coords[0]) for coords in coordinates])

    return polygons

def get_county_name(county_geoid):
    
    with open(county_geojson, encoding="utf-8") as f:
        county_geojson_data = json.load(f)
    for feature in county_geojson_data['features']:
        if feature['properties']['GEOID'] == county_geoid:
            return feature['properties']['COUNTY_STATE_NAME']
        
def get_country_name(country_geoid):
    
    with open(country_geojson, encoding="utf-8") as f:
        country_geojson_data = json.load(f)
    for feature in country_geojson_data['features']:
        if feature['id'] == country_geoid:
            return feature['properties']['name']

def get_geo_id(long, lat, polygons):

    try:
        point = Point(float(long), float(lat))  
    except ValueError:
        logger.warning(f"Invalid coordinates: lat={lat}, long={long}")
        return None  

    for geo_id, polygon in polygons.items():
        try:
            if not polygon.is_valid: 
                polygon = make_valid(polygon)  
                
            if polygon.contains(point):
                return geo_id 
        except Exception as e:
            logger.error(f"Error testing point for {geo_id}: {e}")


def disambiguate_entity_with_coords_gpt(entity_type, entity, sentence, city, state,
                                        county_polygons, state_polygons, country_polygons, 
                                        publisher_state_geoid):

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("GPT disambiguation requires OPENAI_API_KEY.")
    try:
        client = OpenAI(api_key=key)
        prompt = (
            f"The following sentence is from a news article located in {city}, {state}. "
            f"Disambiguate the {entity_type} toponym entity '{entity}' in the sentence: \"{sentence}\". "
            f"Return both the coordinates (in decimal format) and administrative level (county, state, or country).\n"
            f"Format: latitude: <value>, longitude: <value>, type: <county/state/country>"
        )
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200
        )
        response_text = response.choices[0].message.content.strip()

        pattern = r"latitude\s*:\s*([-+]?\d*\.\d+|\d+)\s*,\s*longitude\s*:\s*([-+]?\d*\.\d+|\d+)\s*,\s*type\s*:\s*(county|state|country)"
        match = re.search(pattern, response_text, re.IGNORECASE)

        if not match:
            logger.warning("Could not parse GPT response.")
            return None

        latitude = float(match.group(1))
        longitude = float(match.group(2))
        admin_type = match.group(3).lower()

        if admin_type == 'country':
            geo_id = get_geo_id(longitude, latitude, country_polygons)
            return {
                "latitude": latitude, 
                "longitude": longitude, 
                "ADM": admin_type, 
                "IGL": "national" if geo_id == "USA" else "international", 
                "geoid": geo_id}
        elif admin_type == 'state':
            geo_id = get_geo_id(longitude, latitude, state_polygons)
            return {
                "latitude": latitude, 
                "longitude": longitude, 
                "ADM": admin_type, 
                "IGL": "state" if geo_id == publisher_state_geoid else "national", 
                "geoid": geo_id}
        elif admin_type == 'county':
            state_geo_id = get_geo_id(longitude, latitude, state_polygons)
            county_geo_id = get_geo_id(longitude, latitude, county_polygons)
            if state_geo_id == publisher_state_geoid:
                return {
                    "latitude": latitude, 
                    "longitude": longitude,
                    "ADM": admin_type, 
                    "IGL": "local", 
                    "geoid": county_geo_id}
            else:
                return {
                    "latitude": latitude, 
                    "longitude": longitude, 
                    "ADM": admin_type, 
                    "IGL": "national", 
                    "geoid": county_geo_id}

    except Exception as e:
        logger.error(f"GPT disambiguation failed: {e}")
        return None


def _get_huggingface_client():
    """Create and cache a client for Hugging Face Inference Providers."""
    global _hf_client, _hf_client_token
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face inference requires HF_TOKEN with Inference Providers permission."
        )
    if _hf_client is not None and _hf_client_token == token:
        return _hf_client

    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face backend requires huggingface_hub. "
            "Install the project dependencies before using it."
        ) from exc

    _hf_client = InferenceClient(provider="auto", api_key=token, timeout=120)
    _hf_client_token = token
    return _hf_client



def _extract_json_array(response_text):
    """Extract and parse a JSON array from a model response."""
    text = str(response_text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found in model response: {response_text!r}")

    parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Model response was not a JSON array.")
    return parsed


def _coordinates_to_geo_result(
        latitude, longitude, admin_type,
        county_polygons, state_polygons, country_polygons,
        publisher_state_geoid):
    """Convert coordinates and administrative type to the NLGF result format."""
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return None

    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None

    admin_type = str(admin_type).strip().lower()
    local_types = {
        "city", "town", "village", "municipality", "borough",
        "local", "county", "parish"
    }
    if admin_type in local_types:
        admin_type = "county"

    if admin_type == "country":
        geo_id = get_geo_id(longitude, latitude, country_polygons)
        if not geo_id:
            return None
        igl = "national" if geo_id == "USA" else "international"
    elif admin_type == "state":
        geo_id = get_geo_id(longitude, latitude, state_polygons)
        if not geo_id:
            return None
        igl = "state" if geo_id == publisher_state_geoid else "national"
    elif admin_type == "county":
        geo_id = get_geo_id(longitude, latitude, county_polygons)
        state_geo_id = get_geo_id(longitude, latitude, state_polygons)
        if not geo_id:
            return None
        igl = "local" if state_geo_id == publisher_state_geoid else "national"
    else:
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "ADM": admin_type,
        "IGL": igl,
        "geoid": geo_id,
    }


def _build_unique_toponym_items(toponym_entities):
    """Build unique entity-context items while preserving a map to all mentions."""
    unique_items = []
    key_to_id = {}

    for entity in toponym_entities:
        entity_text = str(entity.get("entity", "")).strip()
        entity_type = str(entity.get("class", "")).strip()
        sentences = entity.get("context", {}).get("sents", [])
        sentence = str(sentences[0].get("sent", "")).strip() if sentences else ""

        if not entity_text or not sentence:
            continue

        key = (entity_text.lower(), entity_type.lower(), sentence)
        if key in key_to_id:
            continue

        item_id = f"loc_{len(unique_items):04d}"
        key_to_id[key] = item_id
        unique_items.append({
            "id": item_id,
            "entity": entity_text,
            "entity_type": entity_type,
            "sentence": sentence,
        })

    return unique_items, key_to_id


def disambiguate_entities_with_coords_huggingface(
        toponym_entities, city, state,
        county_polygons, state_polygons, country_polygons,
        publisher_state_geoid, model_id=DEFAULT_HUGGINGFACE_MODEL,
        batch_size=25):
    """Resolve all unique article toponyms in batched Hugging Face requests."""
    if not toponym_entities:
        return {}, {}

    client = _get_huggingface_client()
    model_id = HUGGINGFACE_MODEL_ALIASES.get(model_id, model_id)
    unique_items, key_to_id = _build_unique_toponym_items(toponym_entities)

    result_by_id = {item["id"]: None for item in unique_items}
    if not unique_items:
        return result_by_id, key_to_id

    for start in range(0, len(unique_items), batch_size):
        batch = unique_items[start:start + batch_size]
        request_payload = json.dumps(batch, ensure_ascii=False, indent=2)

        prompt = f"""
The following place-name candidates were extracted from a news article
published in {city}, {state}.

Some candidates may not be geographic locations. For example, abbreviations
such as "AI" may have been incorrectly identified as places.

Resolve every item using its sentence context.

For each input item, return exactly one output object with the same "id".

If the candidate is not a geographic place, return:
{{"id":"<same id>","is_location":false,"latitude":null,"longitude":null,"type":null}}

If the candidate is a geographic place, return:
{{"id":"<same id>","is_location":true,"latitude":<decimal>,"longitude":<decimal>,"type":"<county|state|country>"}}

Use "county" for cities, towns, villages, counties, municipalities, and other local places.
Use "state" only for a state or equivalent first-level region.
Use "country" only for a country.
Return valid JSON only, as one JSON array, with one object for every input item.

Input items:
{request_payload}
""".strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You accurately resolve geographic place names. "
                    "You always return valid JSON and preserve every input ID."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        max_tokens = min(4096, max(512, len(batch) * 120))
        print(
            f"Sending Hugging Face batch {start // batch_size + 1} "
            f"for {len(batch)} unique toponym candidates."
        )

        try:
            response = client.chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            response_text = response.choices[0].message.content.strip()
            parsed_items = _extract_json_array(response_text)
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face batch disambiguation failed for {len(batch)} "
                f"entities using model {model_id!r}: {exc}"
            ) from exc

        expected_ids = {item["id"] for item in batch}
        for parsed_item in parsed_items:
            if not isinstance(parsed_item, dict):
                continue

            item_id = str(parsed_item.get("id", "")).strip()
            if item_id not in expected_ids:
                continue

            is_location = parsed_item.get("is_location")
            if isinstance(is_location, str):
                is_location = is_location.strip().lower() == "true"
            else:
                is_location = bool(is_location)

            if not is_location:
                result_by_id[item_id] = None
                continue

            result_by_id[item_id] = _coordinates_to_geo_result(
                latitude=parsed_item.get("latitude"),
                longitude=parsed_item.get("longitude"),
                admin_type=parsed_item.get("type"),
                county_polygons=county_polygons,
                state_polygons=state_polygons,
                country_polygons=country_polygons,
                publisher_state_geoid=publisher_state_geoid,
            )

    resolved = sum(value is not None for value in result_by_id.values())
    print(f"Resolved {resolved} of {len(result_by_id)} unique candidates.")
    return result_by_id, key_to_id


def disambiguate_entities_with_coords(
        toponym_entities, city, state,
        county_polygons, state_polygons, country_polygons,
        publisher_state_geoid, backend="huggingface",
        huggingface_model=DEFAULT_HUGGINGFACE_MODEL):
    """Batch dispatcher for toponym disambiguation."""
    if backend in {"huggingface", "llama"}:
        return disambiguate_entities_with_coords_huggingface(
            toponym_entities=toponym_entities,
            city=city,
            state=state,
            county_polygons=county_polygons,
            state_polygons=state_polygons,
            country_polygons=country_polygons,
            publisher_state_geoid=publisher_state_geoid,
            model_id=huggingface_model,
        )
    raise ValueError("Batch disambiguation currently supports only Hugging Face.")


def disambiguate_entity_with_coords_huggingface(
        entity_type, entity, sentence, city, state,
        county_polygons, state_polygons, country_polygons,
        publisher_state_geoid, model_id=DEFAULT_HUGGINGFACE_MODEL):
    """Resolve one toponym through Hugging Face Inference Providers."""
    client = _get_huggingface_client()
    model_id = HUGGINGFACE_MODEL_ALIASES.get(model_id, model_id)
    try:
        prompt = (
            f"The sentence is from a news article published in {city}, {state}. "
            f"Resolve the {entity_type} place name '{entity}' in this sentence: "
            f"\"{sentence}\". Return only one line in exactly this format: "
            "latitude: <decimal>, longitude: <decimal>, "
            "type: <county/state/country>."
        )
        messages = [
            {"role": "system", "content": "You accurately resolve geographic place names."},
            {"role": "user", "content": prompt},
        ]
        response = client.chat_completion(
            model=model_id,
            messages=messages,
            max_tokens=80,
        )
        response_text = response.choices[0].message.content.strip()

        pattern = r"latitude\s*:\s*([-+]?\d*\.?\d+)\s*,\s*longitude\s*:\s*([-+]?\d*\.?\d+)\s*,\s*type\s*:\s*(county|state|country)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if not match:
            logger.warning("Could not parse Hugging Face response: %s", response_text)
            return None

        latitude, longitude = float(match.group(1)), float(match.group(2))
        admin_type = match.group(3).lower()
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            logger.warning("Hugging Face model returned invalid coordinates: %s", response_text)
            return None

        if admin_type == "country":
            geo_id = get_geo_id(longitude, latitude, country_polygons)
            igl = "national" if geo_id == "USA" else "international"
        elif admin_type == "state":
            geo_id = get_geo_id(longitude, latitude, state_polygons)
            igl = "state" if geo_id == publisher_state_geoid else "national"
        else:
            geo_id = get_geo_id(longitude, latitude, county_polygons)
            state_geo_id = get_geo_id(longitude, latitude, state_polygons)
            igl = "local" if state_geo_id == publisher_state_geoid else "national"

        if not geo_id:
            logger.warning("Model coordinates did not map to a known %s", admin_type)
            return None
        return {
            "latitude": latitude,
            "longitude": longitude,
            "ADM": admin_type,
            "IGL": igl,
            "geoid": geo_id,
        }
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face disambiguation failed for model '{model_id}': {exc}"
        ) from exc


def disambiguate_entity_with_coords(
        entity_type, entity, sentence, city, state,
        county_polygons, state_polygons, country_polygons,
        publisher_state_geoid, backend="huggingface",
        huggingface_model=DEFAULT_HUGGINGFACE_MODEL):
    """Dispatch to the requested toponym-disambiguation backend."""
    if backend in {"huggingface", "llama"}:
        return disambiguate_entity_with_coords_huggingface(
            entity_type, entity, sentence, city, state,
            county_polygons, state_polygons, country_polygons,
            publisher_state_geoid, model_id=huggingface_model,
        )
    if backend == "gpt":
        return disambiguate_entity_with_coords_gpt(
            entity_type, entity, sentence, city, state,
            county_polygons, state_polygons, country_polygons,
            publisher_state_geoid,
        )
    raise ValueError("backend must be either 'huggingface' or 'gpt'")


def calculate_geoid_scores(entities):

    counts = {}

    filtered_entities = [e for e in entities if 'geoid' in e and e['geoid']]

    for idx, entity in enumerate(filtered_entities):
        entity_type = entity['class']
        geo_id = entity['geoid']
        ADM = entity.get('ADM')
        IGL = entity.get('IGL')
        logging.debug(f"Geoid: {geo_id}, Type: {IGL}")

        if geo_id:
            logging.debug("Started calculating counts...")
            if geo_id not in counts:
                counts[geo_id] = {'title': 0, 
                                  'GPE': 0, 
                                  'leading_toponym': 0, 
                                  'count':0,
                                  'ADM': ADM,
                                  'IGL': IGL}

            counts[geo_id]['count'] += 1
            
            if entity['is_from_title']:
                counts[geo_id]['title'] += 1

            if entity_type == 'GPE':
                counts[geo_id]['GPE'] += 1
                
            if idx < 5:
                counts[geo_id]['leading_toponym'] += 1

        logging.debug(f"Final Geoid: {geo_id}, Type: {IGL}, Score: {counts[geo_id]}")
        
    return counts


def get_features(link, publisher_longitude, publisher_latitude,
                 disambiguation_backend="huggingface",
                 huggingface_model=DEFAULT_HUGGINGFACE_MODEL):

    article = Article(link)
    article.download()
    article.parse()

    title = article.title
    content = article.text
    if not content or not title:
        logger.warning("Missing content or title for link: %s", link)

    nlp = spacy.load("en_core_web_sm")
    toponym_entities = parallel_ner(nlp, title, content)

    county_polygons = load_geojson(county_geojson, county_geojson, state_geojson)
    state_polygons = load_geojson(state_geojson, county_geojson, state_geojson)
    country_polygons = load_geojson(country_geojson, county_geojson, state_geojson)

    publisher_county_geoid = get_geo_id(
        publisher_longitude, publisher_latitude, county_polygons
    )
    publisher_state_geoid = get_geo_id(
        publisher_longitude, publisher_latitude, state_polygons
    )

    county_name = get_county_name(publisher_county_geoid)
    if not county_name:
        raise RuntimeError(
            "Could not determine publisher county from "
            f"longitude={publisher_longitude}, latitude={publisher_latitude}."
        )

    city, state = [part.strip() for part in county_name.split(",", maxsplit=1)]
    print(f"Detected {len(toponym_entities)} raw toponym mentions.")

    if disambiguation_backend in {"huggingface", "llama"}:
        result_by_id, key_to_id = disambiguate_entities_with_coords(
            toponym_entities=toponym_entities,
            city=city,
            state=state,
            county_polygons=county_polygons,
            state_polygons=state_polygons,
            country_polygons=country_polygons,
            publisher_state_geoid=publisher_state_geoid,
            backend=disambiguation_backend,
            huggingface_model=huggingface_model,
        )

        entities = []
        for entity in toponym_entities:
            entity_text = str(entity.get("entity", "")).strip()
            entity_type = str(entity.get("class", "")).strip()
            sentences = entity.get("context", {}).get("sents", [])
            sentence = str(sentences[0].get("sent", "")).strip() if sentences else ""
            key = (entity_text.lower(), entity_type.lower(), sentence)
            item_id = key_to_id.get(key)
            geo_id_info = result_by_id.get(item_id) if item_id else None
            if geo_id_info:
                entity.update(geo_id_info.copy())
            entities.append(entity)
    else:
        entities = []
        for entity in toponym_entities:
            geo_id_info = disambiguate_entity_with_coords(
                entity['class'], entity['entity'],
                entity['context']['sents'][0]['sent'],
                city, state,
                county_polygons, state_polygons, country_polygons,
                publisher_state_geoid,
                backend=disambiguation_backend,
                huggingface_model=huggingface_model,
            )
            if geo_id_info:
                entity.update(geo_id_info)
            entities.append(entity)

    toponym_scores = calculate_geoid_scores(entities)
    features = extract_features(toponym_scores)
    return features, toponym_scores

def generate_dataset(jsonl_file_path, label, csv_file_path, county_polygons, state_polygons,
                     country_polygons, disambiguation_backend="huggingface",
                     huggingface_model=DEFAULT_HUGGINGFACE_MODEL):
    data = []
    seen_links = set()
    existing_links = set()

    if os.path.exists(csv_file_path):
        try:
            existing_df = pd.read_csv(csv_file_path)
            existing_links = set(existing_df['link'].dropna())
            print(f"Loaded {len(existing_links)} existing links from CSV.")
        except Exception as e:
            logger.warning(f"Failed to read existing CSV: {e}")

    nlp = spacy.load("en_core_web_sm")
    sniffer = StorySniffer()

    print(f"Opening and reading data from {jsonl_file_path}...")
    with gzip.open(jsonl_file_path, 'rt', encoding='utf-8') as f:
        total_lines = sum(1 for _ in gzip.open(jsonl_file_path, 'rt', encoding='utf-8'))
        f.seek(0)

        for idx, line in enumerate(tqdm(f, total=total_lines), start=1):
            try:
                json_obj = json.loads(line)
                link = json_obj.get('link')
                print(f"[{idx}/{total_lines}] Processing link: {link}")

                if link in seen_links or link in existing_links:
                    print(f"Skipping duplicate link: {link}")
                    continue
                seen_links.add(link)

                is_news_article = sniffer.guess(link)
                print(f"Link is news article: {is_news_article}")
                if not is_news_article:
                    continue

                location = json_obj.get('location')
                title = json_obj.get('title')

                article = Article(link)
                article.download()
                article.parse()

                if not title:
                    title = article.title
                logger.debug(f"Article title: {title}")

                content = article.text
                if not content or not title:
                    logger.warning(f"Missing content or title for link: {link}")
                    continue

                toponym_entities = parallel_ner(nlp, title, content)

                publisher_state_geoid = get_geo_id(location.get('longitude'), location.get('latitude'), state_polygons)

                if disambiguation_backend in {"huggingface", "llama"}:
                    result_by_id, key_to_id = disambiguate_entities_with_coords(
                        toponym_entities=toponym_entities,
                        city=location.get('city'),
                        state=location.get('state'),
                        county_polygons=county_polygons,
                        state_polygons=state_polygons,
                        country_polygons=country_polygons,
                        publisher_state_geoid=publisher_state_geoid,
                        backend=disambiguation_backend,
                        huggingface_model=huggingface_model,
                    )

                    entities = []
                    for entity in toponym_entities:
                        entity_text = str(entity.get("entity", "")).strip()
                        entity_type = str(entity.get("class", "")).strip()
                        sentences = entity.get("context", {}).get("sents", [])
                        sentence = str(sentences[0].get("sent", "")).strip() if sentences else ""
                        key = (entity_text.lower(), entity_type.lower(), sentence)
                        item_id = key_to_id.get(key)
                        geo_id_info = result_by_id.get(item_id) if item_id else None
                        if geo_id_info:
                            entity.update(geo_id_info.copy())
                        entities.append(entity)
                else:
                    entities = []
                    for entity in toponym_entities:
                        geo_id_info = disambiguate_entity_with_coords(
                            entity['class'], entity['entity'],
                            entity['context']['sents'][0]['sent'],
                            location.get('city'), location.get('state'),
                            county_polygons, state_polygons, country_polygons,
                            publisher_state_geoid,
                            backend=disambiguation_backend,
                            huggingface_model=huggingface_model,
                        )
                        if geo_id_info:
                            entity.update(geo_id_info)
                        entities.append(entity)

                counts = calculate_geoid_scores(entities)
                features = extract_features(counts)

                row = {
                    'id': str(uuid.uuid4()),
                    'link': link,
                    'location': location,
                    'title': title,
                    'content': content,
                    'toponym_entities': entities,
                    'publisher_state_geoid': publisher_state_geoid,
                    'toponym_scores': counts,
                    'label': label
                }
                row.update(features)
                data.append(row)

            except Exception as e:
                logger.error(f"Failed to process record {idx}: {e}", exc_info=True)

    if data:
        try:
            new_df = pd.DataFrame(data)
            if os.path.exists(csv_file_path):
                existing_df = pd.read_csv(csv_file_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(csv_file_path, index=False)
                print(f"Appended {len(new_df)} rows to existing CSV: {csv_file_path}")
            else:
                new_df.to_csv(csv_file_path, index=False)
                print(f"Saved {len(new_df)} rows to new CSV: {csv_file_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")


def igl_matches_label(igl_value, label):
    if not igl_value or not label:
        return False
    igl_value = str(igl_value).strip().lower()
    label = str(label).strip().lower()
    if igl_value == label:
        return True
    if igl_value == "county" and label == "local":
        return True
    if igl_value == "country" and label == "national":
        return True
    return False


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


def get_geo_focus_label(toponym_scores, label, threshold=0.25, verbose=False):
    if label == 'none':
        return ['none']

    if not isinstance(toponym_scores, dict) or len(toponym_scores) == 0 or not label:
        return ['none']

    filtered = {k: v for k, v in toponym_scores.items() if igl_matches_label(v.get("IGL"), label)}
    if not filtered:
        if verbose: print(f"   ⚠️ No IGL match for label '{label}'.")
        return ['none']

    raw_scores = {k: v.get("title", 0) + v.get("GPE", 0) +
                     v.get("leading_toponym", 0) + v.get("count", 0)
                  for k, v in filtered.items()}
    total = sum(raw_scores.values())
    if total == 0:
        if verbose: print("   ⚠️ Total score = 0 after filtering.")
        return ['none']

    relative_scores = {k: s / total for k, s in raw_scores.items()}

    if verbose:
        print(f"   Label: {label}, Threshold: {threshold}")
        print(f"   Raw Scores: {raw_scores}")
        print(f"   Relative Scores: {relative_scores}")

    selected = [k for k, val in relative_scores.items() if val >= threshold]


    if not selected:
        if verbose: print("   ⚠️ No items exceeded threshold, returning ['None'].")
        return ['none']

    if verbose: print(f"   ✅ Selected: {selected}\n")
    return selected


def evaluate(df, mlb, threshold, label):
    y_true = df['geo_focus_label'].tolist()
    y_pred = [get_geo_focus_label(scores, label, threshold)
              for scores, label in zip(df['toponym_scores'], df[label])]

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    precision = precision_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    return precision, recall, f1, y_pred


def get_best_threshold(train_df, mlb):
    thresholds = np.linspace(0.05, 0.5, 10)
    train_results = []
    print("\nThreshold tuning on training set...")
    for th in thresholds:
        p, r, f1, _ = evaluate(train_df, mlb, th, 'label')
        print(f"   Threshold={round(th,2)} → Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        train_results.append({"threshold": round(th, 2), "precision": p, "recall": r, "f1": f1})

    train_metrics_df = pd.DataFrame(train_results)
    best_threshold = train_metrics_df.loc[train_metrics_df['f1'].idxmax(), 'threshold']
    print("\nBest threshold from training set:", best_threshold)
    return best_threshold

def extract_features(counts):

    def normalize_igl(v):
        if v is None:
            return 'international'
        v = str(v).lower()
        if v in ('country', 'national', 'nat'):
            return 'national'
        if v in ('state', 'province', 'adm1'):
            return 'state'
        if v in ('local', 'adm2', 'county', 'municipality'):
            return 'local'
        if 'international' in v:
            return 'international'
        return 'international'

    if isinstance(counts, list):
        newd = {}
        for i, item in enumerate(counts):
            if isinstance(item, dict):
                if len(item) == 1 and not any(k in ('title', 'GPE', 'count', 'IGL') for k in item.keys()):
                    for k, v in item.items():
                        newd[k] = v
                else:
                    newd[str(i)] = item
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                newd[item[0]] = item[1]
            else:
                newd[str(i)] = item
        counts = newd

    if not isinstance(counts, dict):
        try:
            counts = ast.literal_eval(str(counts))
        except Exception:
            counts = {}

    items = list(counts.items())

    def safe_get(info, key, default=0):
        try:
            if isinstance(info, dict):
                return info.get(key, default)
            if isinstance(info, (list, tuple)):
                return default
            return default
        except Exception:
            return default

    def sum_by_igl(attr, target_igl):
        s = 0
        for _, score_info in items:
            igl_val = normalize_igl(safe_get(score_info, 'IGL', safe_get(score_info, 'ADM', None)))
            if igl_val == target_igl:
                s += int(safe_get(score_info, attr, 0) or 0)
        return s

    def uniq_count(target_igl):
        return sum(1 for geo_id, score_info in items if normalize_igl(safe_get(score_info, 'IGL', safe_get(score_info, 'ADM', None))) == target_igl)

    title_topo_cnt_intl = sum_by_igl('title', 'international')
    title_topo_cnt_national = sum_by_igl('title', 'national')
    title_topo_cnt_state = sum_by_igl('title', 'state')
    title_topo_cnt_local = sum_by_igl('title', 'local')

    intl_igl_cnt = sum_by_igl('GPE', 'international')
    national_igl_cnt = sum_by_igl('GPE', 'national')
    state_igl_cnt = sum_by_igl('GPE', 'state')
    local_igl_cnt = sum_by_igl('GPE', 'local')

    leading_topo_intl_igl_cnt = sum_by_igl('leading_toponym', 'international')
    leading_topo_national_igl_cnt = sum_by_igl('leading_toponym', 'national')
    leading_topo_state_igl_cnt = sum_by_igl('leading_toponym', 'state')
    leading_topo_local_igl_cnt = sum_by_igl('leading_toponym', 'local')

    international_mentions = sum_by_igl('count', 'international')
    country_mentions = sum_by_igl('count', 'national')
    state_mentions = sum_by_igl('count', 'state')
    county_mentions = sum_by_igl('count', 'local')

    uniq_intl_igl = uniq_count('international')
    uniq_national_igl = uniq_count('national')
    uniq_state_igl = uniq_count('state')
    uniq_local_igl = uniq_count('local')

    return {
        "title_topo_cnt_intl": int(title_topo_cnt_intl),
        "title_topo_cnt_national": int(title_topo_cnt_national),
        "title_topo_cnt_state": int(title_topo_cnt_state),
        "title_topo_cnt_local": int(title_topo_cnt_local),

        "intl_igl_cnt": int(intl_igl_cnt),
        "national_igl_cnt": int(national_igl_cnt),
        "state_igl_cnt": int(state_igl_cnt),
        "local_igl_cnt": int(local_igl_cnt),

        "leading_topo_intl_igl_cnt": int(leading_topo_intl_igl_cnt),
        "leading_topo_national_igl_cnt": int(leading_topo_national_igl_cnt),
        "leading_topo_state_igl_cnt": int(leading_topo_state_igl_cnt),
        "leading_topo_local_igl_cnt": int(leading_topo_local_igl_cnt),

        "international_mentions": int(international_mentions),
        "country_mentions": int(country_mentions),
        "state_mentions": int(state_mentions),
        "county_mentions": int(county_mentions),

        "uniq_intl_igl": int(uniq_intl_igl),
        "uniq_national_igl": int(uniq_national_igl),
        "uniq_state_igl": int(uniq_state_igl),
        "uniq_local_igl": int(uniq_local_igl)
    }


def evaluate_geo_focus(df):

    df = df.copy()
    def safe_literal_eval(v):
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except Exception:
                try:
                    import json
                    return json.loads(v)
                except Exception:
                    return v
        return v

    if 'geo_focus_label' in df.columns:
        df['geo_focus_label'] = df['geo_focus_label'].apply(safe_literal_eval)

    if 'toponym_scores' in df.columns:
        df['toponym_scores'] = df['toponym_scores'].apply(safe_literal_eval)

    print("Dataset loaded, total rows:", len(df))

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    all_geo_labels = set(get_all_geo_labels(county_geojson, state_geojson, country_geojson))
    all_geo_labels.add('none')

    mlb = MultiLabelBinarizer(classes=list(all_geo_labels))
    mlb.fit(list(all_geo_labels))

    best_threshold = get_best_threshold(train_df, mlb)

    p, r, f1, test_predictions_full = evaluate(test_df, mlb, best_threshold, 'predicted_label')

    print("\nFinal Test Set Evaluation")
    print("Threshold:", best_threshold)
    print("Precision:", p)
    print("Recall:", r)
    print("F1:", f1)

    df_all = test_df.copy()
    df_all['predicted_geo_focus_label'] = test_predictions_full
    return df_all
