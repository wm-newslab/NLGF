from geo_focus_utils import load_geojson
from geo_focus_utils import generate_dataset

county_geojson = 'data/resources/county.geojson'
state_geojson = 'data/resources/state-us.geojson'
country_geojson = 'data/resources/countries.geojson'

county_polygons = load_geojson(county_geojson, county_geojson, state_geojson)
state_polygons = load_geojson(state_geojson, county_geojson, state_geojson)
country_polygons = load_geojson(country_geojson, county_geojson, state_geojson)

data_file = '<3dlnews_data.jsonl.gz>'
csv_file = 'data/geo_focus_data.csv'
label = '<label;Local,State,National,International,None>'

generate_dataset(data_file, label, csv_file, county_polygons, state_polygons, country_polygons)