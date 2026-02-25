import os
import gzip
import json

input_folder = '<input_path>'  
output_file = '<output_file>'  
target_word = '<keyword>'  

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in os.listdir(input_folder):
        if filename.endswith('.gz'):
            filepath = os.path.join(input_folder, filename)
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if 'link' in obj and target_word in obj['link']:
                            outfile.write(json.dumps(obj) + '\n')
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON in {filename}")
