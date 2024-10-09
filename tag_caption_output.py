#!/usr/bin/env python3
import json
import requests
import signal
import sys
import shutil
import time
import os

tagged_data = {}

PATH_CAPTIONED = 'images_captioned.json'
PATH_CAPTIONED_BACKUP = 'images_captioned.json'
PATH_CAPTIONED_TAGGED = 'images_captioned_tagged.json'
PATH_CAPTIONED_TAGGED_BACKUP = 'images_captioned_tagged_backup.json'
PATH_PROMPT_TEMPLATE = 'tagging_prompt_template.txt'
SERVER = 'bestiary:5000'

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
    with open(PATH_PROMPT_TEMPLATE, 'r') as f:
        return f.read()

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
            "max_length": 75,
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
            "stop_sequence": ["### Instruction:", "### Response:", "\n### "],
            "use_default_badwordsids": False,
            "bypass_eos": False
        }

        response = requests.post(
            'http://SERVER/api/v1/generate',
            headers={'Content-Type': 'application/json'},
            json=payload,
        )

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} for item {key}")
            return None

        response = response.json()['results']
        text = response[0]['text']

        start_idx = text.find('[')
        end_idx = text.find(']')

        # Extract the tags
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            tags = text[start_idx+1:end_idx].replace('"', '').split(', ')
            item['tags'] = tags
            print(f"Tags for item {key}: {item['tags']}")
        else:
            print(f"Could not parse tags for item {key}")

        # Extract "spicy: n" where n is a number from 1 to 10
        spicy_idx = text.find('spicy:')
        if spicy_idx != -1:
            spicy = text[spicy_idx+7:spicy_idx+8]
            item['spicy'] = spicy
            print(f"Spicy for item {key}: {item['spicy']}")

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

    for key, item in data.items():
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
