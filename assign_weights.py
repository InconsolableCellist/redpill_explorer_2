#!/usr/bin/env python3
import json
import random
import requests
import time
import os

# Path to the tag pairs JSON file
tag_pairs_path = 'data/tag_pairs.json'
weights_output_path = 'data/tag_pairs_with_weights.json'

# Load tag pairs and prompt template
def load_data():
    with open(tag_pairs_path, 'r') as f:
        tag_pairs = json.load(f)

    # If there's already a file with weights, load it to update
    if os.path.exists(weights_output_path):
        with open(weights_output_path, 'r') as f:
            tag_pairs_with_weights = json.load(f)
    else:
        tag_pairs_with_weights = {}

    return tag_pairs, tag_pairs_with_weights

# Save the tag pairs with updated weights
def save_data(tag_pairs_with_weights):
    with open(weights_output_path, 'w') as f:
        json.dump(tag_pairs_with_weights, f, indent=2)
    print(f"Data saved to {weights_output_path}")

# Load the LLM prompt template
def load_prompt_template():
    with open('tag_weights_prompt_template.txt', 'r') as f:
        return f.read()

# Function to check if a tag pair already has a weight assigned
def has_weight(tag1, tag2, weighted_pairs):
    return (tag1 in weighted_pairs and tag2 in weighted_pairs[tag1]) or (tag2 in weighted_pairs and tag1 in weighted_pairs[tag2])

# Function to generate a prompt with random tag pairs
def generate_prompt(tag_pairs, weighted_pairs, chunk_size=10):
    unweighted_pairs = []
    
    # Iterate over the dictionary of tag pairs and collect the ones without weights
    for tag1, pairs in tag_pairs.items():
        for tag2 in pairs:
            if not has_weight(tag1, tag2, weighted_pairs):
                unweighted_pairs.append((tag1, tag2))

    if not unweighted_pairs:
        return None, None

    # Select a random chunk of pairs
    random_pairs = random.sample(unweighted_pairs, min(chunk_size, len(unweighted_pairs)))
    tag_pairs_str = ',\n'.join([json.dumps({"tag1": p[0], "tag2": p[1]}) for p in random_pairs])
    prompt_template = load_prompt_template()
    prompt = prompt_template.replace('%TAG_PAIRS%', tag_pairs_str)
    return prompt, random_pairs

# Function to parse the response from the LLM
def parse_response(response_text, selected_pairs, weighted_pairs_dict):
    try:
        # Extract the JSON part from the response
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1  # Ensure it includes the closing bracket
        if start_idx == -1 or end_idx == -1:
            print("Error: Couldn't find JSON structure in response.")
            return []

        json_part = response_text[start_idx:end_idx]
        parsed_weights = json.loads(json_part)

        for pw in parsed_weights:
            tag1 = pw['tag1']
            tag2 = pw['tag2']
            weight = pw['weight']

            # Update the dictionary
            if tag1 not in weighted_pairs_dict:
                weighted_pairs_dict[tag1] = {}
            weighted_pairs_dict[tag1][tag2] = weight

    except Exception as e:
        print(f"Error parsing response: {e}")

# Main processing loop
def main():
    tag_pairs, tag_pairs_with_weights = load_data()

    # Continue looping until all pairs have weights
    while True:
        prompt, selected_pairs = generate_prompt(tag_pairs, tag_pairs_with_weights, chunk_size=20)
        if not prompt:
            print("All tag pairs have weights assigned.")
            break

        print(f"Sending request for {len(selected_pairs)} pairs...")
        response_text = send_request_to_llm(prompt)

        if response_text:
            parse_response(response_text, selected_pairs, tag_pairs_with_weights)
            save_data(tag_pairs_with_weights)
        else:
            print("No response from LLM.")
        
        # Sleep to avoid overwhelming the LLM
        time.sleep(5)

    print("All tag pairs have been processed.")

def send_request_to_llm(prompt):
    payload = {
        "n": 1,
        "max_context_length": 8192,
        "max_length": 750,
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
        "stop_sequence": ["### END"],
        "use_default_badwordsids": False,
        "bypass_eos": False
    }

    try:
        response = requests.post('http://bestiary:5000/api/v1/generate', headers={'Content-Type': 'application/json'}, json=payload)
        if response.status_code == 200:
            return response.json()['results'][0]['text']
        else:
            print(f"Error: LLM returned status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error sending request: {e}")
        return None

if __name__ == '__main__':
    main()
