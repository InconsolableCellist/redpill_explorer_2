#!/usr/bin/env python
import os
import faiss
import numpy as np
import sqlite3
import pickle

FAISS_INDEX_PATH = 'faiss_index.bin'
INDEX_HASH_KEYS_PATH = 'index_hash_keys.pkl'
DATABASE_PATH = 'image_data.db'
def rebuild_faiss_index():
    global index, index_hash_keys, conn, cursor

    # Step 1: Delete old FAISS index and index_hash_keys.pkl if they exist
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
        print(f"Deleted old FAISS index at {FAISS_INDEX_PATH}")
    else:
        print(f"No existing FAISS index found at {FAISS_INDEX_PATH}")

    if os.path.exists(INDEX_HASH_KEYS_PATH):
        os.remove(INDEX_HASH_KEYS_PATH)
        print(f"Deleted old index_hash_keys at {INDEX_HASH_KEYS_PATH}")
    else:
        print(f"No existing index_hash_keys found at {INDEX_HASH_KEYS_PATH}")

    # Step 2: Connect to the database
    if 'conn' not in globals():
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
    else:
        cursor = conn.cursor()

    # Step 3: Retrieve all embeddings and hashes from the database
    cursor.execute('SELECT hash, embedding FROM image_data WHERE embedding IS NOT NULL')
    data = cursor.fetchall()

    if not data:
        print("No embeddings found in the database.")
        return

    index_hash_keys = []
    embeddings_list = []

    print(f"Found {len(data)} embeddings in the database. Rebuilding FAISS index...")

    for hash_value, embedding_blob in data:
        # Convert embedding blob back to numpy array
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        embeddings_list.append(embedding)
        index_hash_keys.append(hash_value)

    # Convert embeddings to a numpy array
    embeddings_array = np.array(embeddings_list).astype('float32')

    # Step 4: Initialize a new FAISS index
    embedding_size = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(embedding_size)
    print(f"Initialized new FAISS index with embedding size {embedding_size}")

    # Step 5: Add all embeddings to the FAISS index
    index.add(embeddings_array)
    print(f"Added {index.ntotal} embeddings to the FAISS index.")

    # Step 6: Save the FAISS index and index_hash_keys
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(INDEX_HASH_KEYS_PATH, 'wb') as f:
        pickle.dump(index_hash_keys, f)
    print(f"Saved FAISS index to {FAISS_INDEX_PATH} and index_hash_keys to {INDEX_HASH_KEYS_PATH}")

    # Close the database connection if it was opened here
    cursor.close()
    # conn.close()  # Uncomment if you want to close the connection

if __name__ == '__main__':
    rebuild_faiss_index()