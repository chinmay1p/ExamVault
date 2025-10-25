# File: create_vector_db.py
# Purpose: To "index" all questions from data.json into a persistent vector database.
# Run this script *once* after you run processor.py.

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import shutil

JSON_PATH = os.path.join("question_bank_ocr", "data.json")
DB_PATH = "chroma_db"
COLLECTION_NAME = "pyq_collection"
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

def load_question_bank(json_path):
    """Loads the JSON dataset."""
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions.")
        return data
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

def main():
    print("🚀 Starting Vector DB Indexing...")

    # --- NUKING OLD DATABASE ---
    # This ensures we are building the new structure from scratch
    if os.path.exists(DB_PATH):
        print(f"Removing old database at '{DB_PATH}'...")
        shutil.rmtree(DB_PATH)
        print("Old database removed.")
    # -------------------------
    
    # 1. Load the data
    all_questions = load_question_bank(JSON_PATH)
    if not all_questions:
        print("No questions to index. Exiting.")
        return

    # 2. Initialize ChromaDB client (persistent)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 3. Create or get the collection
    print(f"Creating collection: '{COLLECTION_NAME}'")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity
    )
    
    # 4. Initialize the Sentence Transformer model
    print(f"Loading SentenceTransformer model: '{SBERT_MODEL_NAME}'...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    # 5. Prepare data for ChromaDB
    # We will add data in batches
    batch_size = 100
    for i in range(0, len(all_questions), batch_size):
        batch = all_questions[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}...")
        
        # Data to be stored
        documents = []  # The actual question text
        metadatas = []  # All associated metadata (for filtering)
        ids = []        # Unique ID for each question
        embeddings = [] # The vector embeddings
        
        # Prepare lists for this batch
        texts_to_embed = []
        for q_idx, q in enumerate(batch):
            unique_id = f"q_{i + q_idx}"
            
            texts_to_embed.append(q['question_text'])
            documents.append(q['question_text'])
            ids.append(unique_id)
            
            # --- **** MODIFIED METADATA **** ---
            # Base metadata for display
            meta = {
                "source_pdf": q['source_pdf'],
                "year": q['year'],
                "page": q['page'],
                "image_path": q['image_path'],
                "topics_str": ",".join(q['topics']) # For display
            }
            
            # Add each topic as a filterable key
            # e.g., "Neural Networks" -> "topic_neural_networks" = 1
            for topic in q['topics']:
                # Clean the topic name to create a valid metadata key
                topic_key = f"topic_{topic.lower().replace(' ', '_').replace('-', '_')}"
                meta[topic_key] = 1 # Use 1 for presence (like a boolean)
            
            metadatas.append(meta)
            # --- ***************************** ---

        # 6. Generate embeddings for the batch
        batch_embeddings = sbert_model.encode(texts_to_embed, show_progress_bar=False).tolist()
        
        # 7. Add the batch to the collection
        collection.add(
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    print("\n✅ Indexing complete!")
    print(f"Vector DB created at: '{DB_PATH}'")
    print(f"Total questions indexed: {collection.count()}")

if __name__ == "__main__":
    main()