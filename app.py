import os
import chromadb
import sys
from flask import Flask, jsonify, request, send_from_directory, send_file

DB_PATH = "chroma_db"
COLLECTION_NAME = "pyq_collection"
app = Flask(__name__)
collection = None

def initialize_db():
    """Connects to the vector DB and sets the global 'collection'."""
    global collection
    print("Connecting to DB...")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Connected to Vector DB. Found {collection.count()} indexed questions.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR connecting to ChromaDB: {e}")
        print(f"Make sure '{DB_PATH}' exists and you have run 'create_vector_db.py'.")
        sys.exit(1)

# --- Core Search Logic (from your script) ---
def search_by_topic(topic_filter, max_results=1000):
    """Performs a filter-only search using collection.get()."""
    
    # Clean the input topic to match the key we created
    topic_key = f"topic_{topic_filter.lower().replace(' ', '_').replace('-', '_')}"
    
    where_filter = {
        topic_key: {
            "$eq": 1  # Check for the presence of the key
        }
    }

    try:
        results = collection.get(
            where=where_filter,
            limit=max_results 
        )
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        print("This might mean the topic key was not found (no results).")
        # Ensure 'ids' is also returned for consistency, even on error
        return {'documents': [], 'metadatas': [], 'ids': []}

# --- API Endpoint for Searching ---
# This is what your JavaScript will call
@app.route('/search')
def search_api():
    """
    API endpoint to search for questions by topic.
    Takes a URL parameter: /search?topic=SVM
    """
    # Get the topic from the URL parameter
    topic = request.args.get('topic')
    
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
        
    print(f"API: Received search for topic: '{topic}'")
    
    # Run your Python search function
    results = search_by_topic(topic)
    
    # Return the results as JSON
    return jsonify(results)

# --- Routes to Serve Your Website Files ---

@app.route('/')
def serve_index():
    """Serves the main index.html page."""
    return send_file('index.html')

@app.route('/search.html')
def serve_search():
    """Serves the search.html page."""
    return send_file('search.html')

# --- Routes to Serve Static Files (PDFs, Images) ---
# These routes allow your HTML to access files from these folders.

@app.route('/question_bank_ocr/<path:filename>')
def serve_ocr_files(filename):
    """Serves images from the ocr folder."""
    # e.g., /question_bank_ocr/images/ML_2022_...png
    return send_from_directory('question_bank_ocr', filename)

@app.route('/question_paper_bank/<path:filename>')
def serve_pdf_files(filename):
    """Serves PDFs from the paper bank folder."""
    # e.g., /question_paper_bank/ML/ML_2023.pdf
    return send_from_directory('question_paper_bank', filename)


# --- Run the App ---
if __name__ == '__main__':    
    initialize_db()
    print("Starting Flask server at http://localhost:5000")
    app.run(debug=True, port=5000)
