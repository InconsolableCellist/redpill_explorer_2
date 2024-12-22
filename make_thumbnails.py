#!/usr/bin/env python 
import json
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

# Load the JSON file
with open('data/adjusted_data.json', 'r') as f:
    data = json.load(f)

# Function to process a single image
def process_image(hash_key, entry):
    try:
        # get the path with a GET to the server at port 3000 /hash-to-path/:hash
        img_path = requests.get(f"http://localhost:3000/hash-to-path/{hash_key}").json()['path']
        if not os.path.exists(img_path):
            print(f"File {img_path} does not exist.")
            return
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB mode for JPEG
        
        width = 400
        if img.size[0] > width:
            w_percent = (width / float(img.size[0]))
            height = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((width, height), resample=Image.LANCZOS)
        
        # Determine the output path based on the hash
        # Use the first 3 characters of the hash for sharding
        subdirs = [hash_key[0], hash_key[1], hash_key[2]]
        output_dir = os.path.join('public', 'thumbnails', *subdirs)
        
        # Ensure the directories exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the rest of the hash as the filename with .jpg extension
        output_filename = f"{hash_key[3:]}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the thumbnail as JPEG
        img.save(output_path, format='JPEG', quality=85)
    except Exception as e:
        print(f"Error processing image {hash_key}: {e}")

# Process images in threads
def main():
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for hash_key, entry in data.items():
            futures.append(executor.submit(process_image, hash_key, entry))
        
        # Use tqdm to show progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            pass  # You can handle results here if needed

if __name__ == "__main__":
    main()
