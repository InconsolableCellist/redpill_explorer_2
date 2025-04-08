#!/usr/bin/env python3
# server.py

import os
from fastapi import FastAPI, Body, UploadFile, File, Query
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

import pickle
import hashlib
import torch
import faiss
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, Body, UploadFile, File, Query
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import transformers
from typing import List, Optional
import sqlite3

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

LOCAL_MODEL_DIR = "./models/THUDM/cogvlm2-llama3-chat-19B/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

origins = [
    "http://localhost:3000",
]

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 128002
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = 128003

print(f"pad_token_id: {tokenizer.pad_token_id}")
print(f"eos_token_id: {tokenizer.eos_token_id}")

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
    )

num_gpus = torch.cuda.device_count()

max_memory = {
    0: "20.5GiB",
    1: "23GiB"
}

if num_gpus != len(max_memory):
    raise ValueError(f"Number of GPUs ({num_gpus}) does not match the number of specified memory limits ({len(max_memory)}).")

device_map = infer_auto_device_map(
    model=model,
    max_memory=max_memory,
    no_split_module_classes=["CogVLMDecoderLayer"]
)

offload_folder = "/tmp/offload"
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder)

app = FastAPI(
    title="CogVLM2 Multi-GPU Image Captioning API",
    description="This API generates detailed captions for images using CogVLM2, stores embeddings using FAISS, and provides debugging information. It supports multi-GPU inference for large models.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_checkpoint_and_dispatch(
    model=model,
    checkpoint=LOCAL_MODEL_DIR,
    device_map=device_map,
    dtype=TORCH_TYPE,
    offload_folder=offload_folder,
)

model = model.eval()

task_prompt = ("Provide a detailed caption that fully describes the image. Be sure to describe any text the image "
               "contains. If it contains NSFW content, describe it in detail and include a NSFW specifier. If the "
               "image contains people of identifiable races, be sure to mention those, including but not limited to "
               "black, African black, white, Asian, Hispanic, Jewish, or Arab. If the image contains animals, "
               "describe them in detail. Attempt to describe any jokes. Attempt to describe any memes. If the "
               "image seems controversial or political, attempt to describe that in detail, including why it may be "
               "offensive. If it does not contain any of these elements don't bother mentioning their absence.")

index = None
index_to_metadata = {}
index_hash_keys = []

FAISS_INDEX_PATH = "faiss_index.bin"
INDEX_HASH_KEYS_PATH = 'index_hash_keys.pkl'

conn = sqlite3.connect('image_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_data (
        hash TEXT PRIMARY KEY,
        filename TEXT,
        description TEXT,
        embedding BLOB
    )
''')



def save_faiss_index():
    global index, index_hash_keys
    if index is not None:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(INDEX_HASH_KEYS_PATH, 'wb') as f:
            pickle.dump(index_hash_keys, f)
        print(f"Saved FAISS index with {index.ntotal} embeddings and index_hash_keys with {len(index_hash_keys)} entries.")
        conn.commit()

def load_faiss_index():
    global index, index_hash_keys
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"Loaded FAISS index with {index.ntotal} embeddings.")
        if os.path.exists(INDEX_HASH_KEYS_PATH):
            with open(INDEX_HASH_KEYS_PATH, 'rb') as f:
                index_hash_keys = pickle.load(f)
            print(f"Loaded index_hash_keys with {len(index_hash_keys)} entries.")
        else:
            print("index_hash_keys file not found. Initializing empty list.")
            index_hash_keys = []
    else:
        print("FAISS index file not found. Initializing empty index and index_hash_keys.")
        index = None
        index_hash_keys = []

load_faiss_index()

def generate_text_embedding(text):
    try:
        # Prepare input for the model without images
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=text,
            history=[],
            images=None,  # No images provided
            template_version='chat'
        )

        # Create tensors from input
        input_ids = input_by_model['input_ids'].unsqueeze(0).to(DEVICE)
        token_type_ids = input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE)
        attention_mask = input_by_model['attention_mask'].unsqueeze(0).to(DEVICE)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Calculate the mean of the last hidden state for embedding
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0).to(torch.float32).cpu().numpy()

        return embedding
    except Exception as e:
        print(f"Error in generate_text_embedding: {e}")
        return None

def generate_caption(image):
    try:
        # Prepare input for the model
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=task_prompt,
            history=[],
            images=[image],  # Use the resized image
            template_version='chat'
        )

        # Create tensors from input
        input_ids = input_by_model['input_ids'].unsqueeze(0).to(DEVICE)
        token_type_ids = input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE)
        attention_mask = input_by_model['attention_mask'].unsqueeze(0).to(DEVICE)
        images_tensor = [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]]

        # Generate caption
        caption_inputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'images': images_tensor,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
        }
        with torch.no_grad():
            # Generate the text response (caption)
            caption_outputs = model.generate(**caption_inputs, **gen_kwargs)
            caption_outputs = caption_outputs[:, input_ids.shape[1]:]
            caption = tokenizer.decode(caption_outputs[0])
            caption = caption.split(tokenizer.eos_token)[0].strip()

        print(f"Generated caption: {caption}")
        # Generate embeddings separately by calling the model directly
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                images=images_tensor,
                output_hidden_states=True
            )

            # Calculate the mean of the last hidden state for embedding
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0).to(torch.float32).cpu().numpy()
        return caption, embedding
    except Exception as e:
        print(f"Error in generate_caption: {e}")
        return {"error": f"An error occurred: {str(e)}"}, None

def process_image(image_data, filename, insert_embeddings):
    global index, index_hash_keys, cursor, conn
    results = {}
    db_status = {}
    try:
        hash_value = hashlib.sha256(image_data).hexdigest()

        # Check if image has been processed before
        cursor.execute('SELECT description, embedding FROM image_data WHERE hash=?', (hash_value,))
        row = cursor.fetchone()
        if row:
            description, embedding_blob = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            db_status[hash_value] = "Retrieved from database."
        else:
            image = Image.open(BytesIO(image_data)).convert("RGB")
            description, embedding = generate_caption(image)
            if insert_embeddings:
                if embedding is not None:
                    # Add to FAISS Index
                    if index is None:
                        embedding_size = embedding.shape[0]
                        index = faiss.IndexFlatL2(embedding_size)
                    index.add(np.array([embedding]).astype("float32"))
                    index_hash_keys.append(hash_value)
                    cursor.execute('''
                        INSERT INTO image_data (hash, filename, description, embedding)
                        VALUES (?, ?, ?, ?)
                    ''', (hash_value, filename, description, embedding.tobytes()))
                    conn.commit()
                    db_status[hash_value] = "Successfully added to FAISS and database."
                else:
                    db_status[hash_value] = "Failed to generate embedding."
            else:
                db_status[hash_value] = "Embedding insertion disabled."
        results[hash_value] = {
            "filename": filename,
            "description": description
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        db_status[filename] = f"An error occurred: {str(e)}"
    return results, db_status


@app.post("/caption", summary="Generate Caption for Image")
async def caption_image(
        file: UploadFile = File(None),
        insert_embeddings: bool = Query(True),
        image_paths: Optional[List[str]] = Body(None)
):
    results = {}
    db_status = {}

    if file is not None:
        # Process the uploaded file
        image_data = await file.read()
        res, status = process_image(image_data, file.filename, insert_embeddings)
        results.update(res)
        db_status.update(status)
    elif image_paths is not None:
        # Process images from paths
        for image_path in image_paths:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                res, status = process_image(image_data, image_path, insert_embeddings)
                results.update(res)
                db_status.update(status)
            else:
                db_status[image_path] = "File does not exist."
    else:
        return {"error": "No file uploaded or image paths provided.", "db_status": db_status}

    save_faiss_index()
    return {"results": results, "db_status": db_status}


@app.post("/search", summary="Search images in FAISS index")
async def search_image(
        file: UploadFile = File(None),
        image_path: Optional[str] = Body(None),
        image_hash: Optional[str] = Body(None),
        text: Optional[str] = Body(None),
        k: int = Query(5, description="Number of nearest neighbors to retrieve"),
        store_in_db: bool = Query(True)
):
    global index, index_hash_keys, index_to_metadata

    # Ensure the FAISS index and index_hash_keys are loaded
    if index is None or len(index_hash_keys) == 0:
        load_faiss_index()

    if index is None or index.ntotal == 0:
        print(f"FAISS index is empty. Index total: {index.ntotal if index else 'None'}")
        print(f"index_hash_keys length: {len(index_hash_keys)}")
        return {"error": "FAISS index is empty. Add images with embeddings first."}

    try:
        # Determine the source of the query embedding
        if text is not None:
            # Generate embedding from text
            embedding = generate_text_embedding(text)
            if embedding is None:
                return {"error": "Failed to generate embedding for the provided text."}
        elif file is not None:
            # Process the uploaded image file
            image_data = await file.read()
            filename = file.filename
            # Generate embedding from image
            image = Image.open(BytesIO(image_data)).convert("RGB")
            _, embedding = generate_caption(image)
            if embedding is None:
                return {"error": "Failed to generate embedding for the uploaded image."}
            # Optionally store the embedding and image data
            if store_in_db:
                hash_value = hashlib.sha256(image_data).hexdigest()
                cursor.execute('SELECT embedding FROM image_data WHERE hash=?', (hash_value,))
                row = cursor.fetchone()
                if row is None:
                    # Store in database
                    cursor.execute('''
                        INSERT INTO image_data (hash, filename, description, embedding)
                        VALUES (?, ?, ?, ?)
                    ''', (hash_value, filename, "", embedding.tobytes()))
                    conn.commit()
                    # Add to FAISS index
                    index.add(np.array([embedding]).astype("float32"))
                    index_hash_keys.append(hash_value)
                    save_faiss_index()
        elif image_path is not None:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                filename = image_path
                # Generate embedding from image
                image = Image.open(BytesIO(image_data)).convert("RGB")
                _, embedding = generate_caption(image)
                if embedding is None:
                    return {"error": "Failed to generate embedding for the image at the provided path."}
                # Optionally store the embedding and image data
                if store_in_db:
                    hash_value = hashlib.sha256(image_data).hexdigest()
                    cursor.execute('SELECT embedding FROM image_data WHERE hash=?', (hash_value,))
                    row = cursor.fetchone()
                    if row is None:
                        # Store in database
                        cursor.execute('''
                            INSERT INTO image_data (hash, filename, description, embedding)
                            VALUES (?, ?, ?, ?)
                        ''', (hash_value, filename, "", embedding.tobytes()))
                        conn.commit()
                        # Add to FAISS index
                        index.add(np.array([embedding]).astype("float32"))
                        index_hash_keys.append(hash_value)
                        save_faiss_index()
            else:
                return {"error": f"File does not exist: {image_path}"}
        elif image_hash is not None:
            cursor.execute('SELECT filename, embedding FROM image_data WHERE hash=?', (image_hash,))
            row = cursor.fetchone()
            if row:
                filename, embedding_blob = row
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            else:
                return {"error": f"Image hash not found in database: {image_hash}"}
        else:
            return {"error": "No input provided for search. Please provide text, an image file, image path, or image hash."}

        # Search in the FAISS index
        print(f"Searching FAISS index with {index.ntotal} embeddings and {len(index_hash_keys)} keys.")
        D, I = index.search(np.array([embedding]).astype("float32"), k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            if 0 <= idx < len(index_hash_keys):
                result_hash = index_hash_keys[idx]
                cursor.execute('SELECT filename, description FROM image_data WHERE hash=?', (result_hash,))
                row = cursor.fetchone()
                if row:
                    result_filename, description = row
                else:
                    result_filename, description = None, None
                results.append({
                    "hash": result_hash,
                    "distance": float(dist),
                    "filename": result_filename,
                    "description": description
                })
            else:
                print(f"Warning: Index {idx} is out of bounds for index_hash_keys (length {len(index_hash_keys)}).")

        return {"results": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred: {str(e)}"}

@app.get("/debug", summary="Display model and embedding information")
async def debug_info():
    embedding_size = index.d if index else None
    model_info = {
        "model_name": LOCAL_MODEL_DIR,
        "device": DEVICE,
        "torch_dtype": str(TORCH_TYPE),
        "embedding_size": embedding_size,
        "faiss_index_size": index.ntotal if index else 0
    }
    return model_info

# Start the server with: uvicorn server:app --host 0.0.0.0 --port 8000
