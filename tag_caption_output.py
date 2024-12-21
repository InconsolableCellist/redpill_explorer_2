#!/usr/bin/env python3
import json
from tqdm import tqdm
import requests
import signal
import sys
import shutil
import time
import os

tagged_data = {}

if not os.path.exists('backup'):
    os.makedirs('backup')

PATH_CAPTIONED = os.path.join('data', 'images_captioned.json')
PATH_CAPTIONED_BACKUP = os.path.join('backup', 'images_captioned.json')
PATH_CAPTIONED_TAGGED = os.path.join('data', 'images_captioned_tagged.json')
PATH_CAPTIONED_TAGGED_BACKUP = os.path.join('backup', 'images_captioned_tagged_backup.json')
PATH_PROMPT_TEMPLATE = 'tagging_prompt_template.txt'
SERVER = 'bestiary:5000'
INSTRUCTION = "You're a captioning bot that takes a short summary of an image and must generate a JSON array of around 15 tags. The tags should fully describe the content of the image and break it out into easily searchable categories. Special attention must be paid to the political, social, cultural, race, sex, gender, or controversial content in the captions. A list of tags is provided but you may choose to generate additional tags if it's especially relevant. The tags should be of the form `{\"tag name\": n}` where n is a value 0.0-1.0 that corresponds to how relevant the tag is. After the tags you must append a spiciness rating based on your judgment of the caption, in the form of spicy: `{\"spicy\": n}`, where n is 0.0-1.0. 0.0 would be e.g., a photo of a happy cat. 1.0 would be, e.g., Hitler dancing on the twin towers on 9/11."
AVAILABLE_TAGS = ""
with open('available_tags.txt', 'r') as f:
    AVAILABLE_TAGS = f.read()

def signal_handler(sig, frame):
    print('Interrupted! Saving data...')
    with open(PATH_CAPTIONED_TAGGED, 'w') as outfile:
        json.dump(tagged_data, outfile, indent=2)
    sys.exit(0)

# Backs up both PATH_CAPTIONED and PATH_CAPTIONED_TAGGED if they exist
def backup_files():
    if os.path.exists(PATH_CAPTIONED):
        if (not os.path.exists(PATH_CAPTIONED_BACKUP) or (time.time() - os.path.getmtime(PATH_CAPTIONED_BACKUP)) > 86400):
            shutil.copy(PATH_CAPTIONED, PATH_CAPTIONED_BACKUP)
            print(f'Backup of {PATH_CAPTIONED} created')

    if os.path.exists(PATH_CAPTIONED_TAGGED):
        if (not os.path.exists(PATH_CAPTIONED_TAGGED_BACKUP) or (time.time() - os.path.getmtime(PATH_CAPTIONED_TAGGED_BACKUP)) > 86400):
            shutil.copy(PATH_CAPTIONED_TAGGED, PATH_CAPTIONED_TAGGED_BACKUP)
            print(f'Backup of {PATH_CAPTIONED_TAGGED} created')

def load_prompt_template():
    template = ""
    with open(PATH_PROMPT_TEMPLATE, 'r') as f:
        return f.read().replace('%INSTRUCTION%', INSTRUCTION).replace('%AVAILABLE_TAGS%', AVAILABLE_TAGS)

def load_data():
    if os.path.exists(PATH_CAPTIONED):
        with open(PATH_CAPTIONED, 'r') as f:
            data = json.load(f)
    else:
        print(f'Error: Captioned images at {PATH_CAPTIONED} not found. Please caption_images.py first.')
        sys.exit(1)

    if os.path.exists(PATH_CAPTIONED_TAGGED):
        with open(PATH_CAPTIONED_TAGGED, 'r') as f:
            tagged_data = json.load(f)
    else:
        tagged_data = {}
    return data, tagged_data


def process_item(key, item, prompt_template):
    try:
        caption = item['description']
        prompt = prompt_template.replace('%CAPTION%', caption)

        payload = {
            "n": 1,
            "max_context_length": 128000,
            "max_length": 500,
            "rep_pen": 1.07,
            "temperature": 0.7,
            "top_p": 0.92,
            "top_k": 100,
            "top_a": 0,
            "typical": 1,
            "tfs": 1,
            "rep_pen_range": 320,
            "rep_pen_slope": 0.7,
            "sampler_order": [6, 0, 1, 3, 4, 2, 5],
            "memory": "",
            "trim_stop": True,
            "genkey": "KCPP8126",
            "min_p": 0,
            "dynatemp_range": 0,
            "dynatemp_exponent": 1,
            "smoothing_factor": 0,
            "banned_tokens": [],
            "render_special": False,
            "presence_penalty": 0,
            "logit_bias": {},
            "prompt": prompt,
            "quiet": True,
            "stop_sequence": ["### Instruction:", "### Response:", "###Human:", "### Assistant:", "\n\n"],
            "use_default_badwordsids": False,
            "bypass_eos": False
        }

        response = requests.post(
            f'http://{SERVER}/api/v1/generate',
            headers={'Content-Type': 'application/json'},
            json=payload,
        )

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} for item {key}")
            return None

        response_text = response.json()['results'][0]['text']

        # Find the JSON object in the response.
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1  # Include the closing brace.
        if json_start == -1 or json_end == -1:
            print(f"Could not find JSON in response for item {key}")
            return None

        json_str = response_text[json_start:json_end]
        try:
            response_data = json.loads(json_str)
            # Extract tags and spiciness.
            item['tags'] = {tag['tag_name']: tag['relevance_score'] for tag in response_data['tags']}
            item['spicy'] = response_data['spicy']['spicy']
            print(f"Tags for item {key}: {item['tags']}")
            print(f"Spiciness for item {key}: {item['spicy']}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for item {key}: {e}")
            return None

        return item

    except Exception as e:
        print(f"Exception occurred for item {key}: {e}")
        return None

def main():
    global tagged_data
    signal.signal(signal.SIGINT, signal_handler)
    backup_files()
    prompt_template = load_prompt_template()
    data, tagged_data = load_data()
    save_count = 0

    progress_bar = tqdm(data.items(), total=len(data), desc='Processing items', unit='captioned image')
    # for key, item in data.items():
    for key, item in progress_bar:
        if key in tagged_data and 'tags' in tagged_data[key]:
            continue

        processed_item = process_item(key, item, prompt_template)
        if processed_item:
            tagged_data[key] = processed_item
            save_count += 1

            if save_count % 10 == 0:
                print(f"Saving data... {save_count} new items processed")
                with open(PATH_CAPTIONED_TAGGED, 'w') as outfile:
                    json.dump(tagged_data, outfile, indent=2)
                save_count = 0
        else:
            continue

    # Save at the end
    with open(PATH_CAPTIONED_TAGGED, 'w') as outfile:
        json.dump(tagged_data, outfile, indent=2)

    print(f'Processing complete. Data saved to {PATH_CAPTIONED_TAGGED}')

if __name__ == '__main__':
    main()
