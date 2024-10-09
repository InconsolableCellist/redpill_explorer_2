#!/usr/bin/env python3
import os
import sys
import hashlib
import json
import requests
import signal
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# Constants
#CAPTION_ENDPOINT = "http://mlboy:8000/caption"
CAPTION_ENDPOINT = "http://bestiary:8000/caption"
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
LOCAL_DB_PATH = 'images_captioned.json'
BATCH_SAVE_INTERVAL = 10

# Load the local database of processed files
def load_local_db(db_path):
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    else:
        return {}

# Save the local database of processed files
def save_local_db(db, db_path):
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=4)
    print(f"Saved local database to {db_path}")

# Handle Ctrl+C and save the database on exit
def handle_exit(signum, frame):
    print("\nSaving local database before exit...")
    save_local_db(processed_files, LOCAL_DB_PATH)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# Calculate SHA-256 hash of the file contents
def calculate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Call the /caption endpoint for the given image file
def caption_image(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f)}
        try:
            response = requests.post(CAPTION_ENDPOINT, files=files)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error processing file {file_path}: {e}")
            return None

def process_image(file_path):
    file_hash = calculate_file_hash(file_path)

    # Skip if the image has already been processed
    if file_hash in processed_files:
        #print(f"File already processed: {file_path}")
        return False

    result = caption_image(file_path)
    if result and "results" in result:
        caption_data = result["results"].get(file_hash, {})
        if "description" in caption_data and caption_data["description"]:
            processed_files[file_hash] = {
                "filename": caption_data["filename"],
                "description": caption_data["description"]
            }

            print(f"Captioned: {caption_data['filename']}")
            print(f"Description: {caption_data['description']}")

            return True
        else:
            print(f"Failed to get a valid caption for {file_path}")
    else:
        print(f"Failed to process {file_path}")

    return False

# Get all image files from the provided directory
def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.append(os.path.join(root, file))
    return image_files

# Main function
def main(path):
    global processed_files

    # Load processed images from the local database
    processed_files = load_local_db(LOCAL_DB_PATH)
    print(f"Loaded local database with {len(processed_files)} entries.")

    new_images_processed = False

    # If path is a file, process only that file
    if os.path.isfile(path):
        print(f"Processing single file: {path}")
        if process_image(path):
            new_images_processed = True
    elif os.path.isdir(path):
        print(f"Processing directory: {path}")
        image_files = get_image_files(path)
        print(f"Found {len(image_files)} image files in {path}.")

        progress_bar = tqdm(image_files, desc="Processing images", unit="file")
        save_counter = 0

        for image_file in progress_bar:
            if process_image(image_file):
                new_images_processed = True
                save_counter += 1

            if save_counter >= BATCH_SAVE_INTERVAL:
                save_local_db(processed_files, LOCAL_DB_PATH)
                save_counter = 0  # Reset the counter

    if new_images_processed:
        save_local_db(processed_files, LOCAL_DB_PATH)
    print("Finished processing all images.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caption_images_with_cogvlm2.py <file_or_directory>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: {path} is not a valid file or directory.")
        sys.exit(1)

    main(path)

