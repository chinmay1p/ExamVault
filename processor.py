# File: processor.py (OCR Version)
# Purpose: Scans all PDFs using OCR, extracts questions, and builds a JSON database.

import os
import glob
import json
import re
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import pytesseract
import sys

# --- New Imports for ML Model ---
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

PDF_SOURCE_DIR = "question_papers"
OUTPUT_DIR = "question_bank_ocr"
IMG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_PATH = os.path.join(OUTPUT_DIR, "data.json")

# --- New ML Model Loading ---
# Define model paths (you must have these files ready)
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
XGB_MODEL_PATH = 'xgb_multi_output_model.joblib' 
MLB_PATH = 'multi_label_binarizer.joblib'

print("Loading ML tagging models...")
try:
    # 1. Load Sentence Transformer (will download if not present)
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    
    # 2. Load the trained XGBoost model
    xgb_model = joblib.load(XGB_MODEL_PATH)
    
    # 3. Load the fitted MultiLabelBinarizer
    mlb = joblib.load(MLB_PATH)
    print("✅ ML tagging models loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Could not find model file: {e.filename}")
    print("Please make sure the following files are in the same directory as the script:")
    print(f"  - {XGB_MODEL_PATH}")
    print(f"  - {MLB_PATH}")
    print("Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load ML models. {e}")
    print("Please ensure 'sentence_transformers', 'sklearn', and 'xgboost' are installed.")
    print("Exiting.")
    sys.exit(1)



def extract_metadata_from_filename(pdf_filename):
    """Extracts year and subject from the PDF filename."""
    match = re.search(r"_(Machine_Learning)_(\d{4})", pdf_filename, re.IGNORECASE)
    if match:
        subject = match.group(1).replace('_', ' ')
        year = match.group(2)
        return subject, year
    return "Unknown Subject", "Unknown Year"

# --- REPLACED FUNCTION ---
def get_topic_tags(question_text):
    try:
        # 1. Create embedding for the question text.
        # SBERT model expects a list of strings.
        embedding = sbert_model.encode([question_text])
        
        # 2. Predict the binary matrix [ [0, 1, 0, 1, ...] ]
        pred_binary = xgb_model.predict(embedding)
        
        # 3. Inverse transform to get tag names [ ('TagA', 'TagD') ]
        predicted_tags_tuple = mlb.inverse_transform(pred_binary)
        
        # 4. Convert from list of tuples to a simple list of strings
        if predicted_tags_tuple:
            tags_list = list(predicted_tags_tuple[0])
        else:
            tags_list = []
        
        # 5. Return the list, or "General" if no tags were predicted
        return tags_list if tags_list else ["General"]
    
    except Exception as e:
        print(f"    - ⚠️  Error in get_topic_tags: {e}. Defaulting to 'General'.")
        return ["General"]

# --------------------------

def find_question_starts_from_ocr(ocr_data):
    """Finds the y-coordinates of lines that start questions from OCR data."""
    q_pattern = re.compile(r"^(Q-\d+\.|[A-Z][\).]|\d+\.)", re.IGNORECASE)
    lines = ocr_data.groupby(['block_num', 'line_num'])
    question_starts = []
    for (block, line), words in lines:
        if words.empty or pd.isna(words.iloc[0]['text']):
            continue
        first_word_text = str(words.iloc[0]['text']).strip()
        if q_pattern.match(first_word_text):
            y0 = words['top'].min()
            question_starts.append(y0)
    question_starts.sort()
    return question_starts

def process_pdf_with_ocr(pdf_path):
    """Processes a single PDF file using an OCR pipeline."""
    print(f"-> Processing {os.path.basename(pdf_path)} with OCR...")
    doc = fitz.open(pdf_path)
    subject, year = extract_metadata_from_filename(os.path.basename(pdf_path))
    
    questions_in_pdf = []

    for page_num, page in enumerate(doc):
        # 1. Convert page to a high-res image
        pix = page.get_pixmap(dpi=300)
        page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 2. Run OCR to get a DataFrame of all words and their coordinates
        ocr_data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DATAFRAME)
        ocr_data.dropna(subset=['text'], inplace=True)
        ocr_data = ocr_data[ocr_data.conf > 30] # Filter low-confidence words

        # 3. Find the starting vertical position of each question
        q_starts_y = find_question_starts_from_ocr(ocr_data)
        if not q_starts_y:
            continue

        for i, y0 in enumerate(q_starts_y):
            # 4. Define the bounding box for the question
            page_width, page_height = page_image.size
            y1 = page_height if (i + 1 >= len(q_starts_y)) else q_starts_y[i+1]
            bbox = (0, y0 - 10, page_width, y1 - 10) # Bbox with a small top margin

            # 5. Crop the image to get the question image
            question_image = page_image.crop(bbox)
            
            # 6. Filter the OCR data to get the text for this question
            q_words_df = ocr_data[(ocr_data['top'] >= bbox[1]) & (ocr_data['top'] < bbox[3])]
            question_text = ' '.join(q_words_df['text'].astype(str))
            
            if len(question_text) < 20: continue

            # 7. Save image and store data
            q_title_slug = re.sub(r'\W+', '_', question_text[:30]).lower()
            img_filename = f"{year}_{subject.replace(' ','')}_{page_num+1}_{i}_{q_title_slug}.png"
            img_path = os.path.join(IMG_OUTPUT_DIR, img_filename)
            question_image.save(img_path)

            # This call now uses the new ML-based function
            topics = get_topic_tags(question_text)

            questions_in_pdf.append({
                "subject": subject, "year": year,
                "question_text": question_text,
                "image_path": img_path.replace("\\", "/"),
                "topics": topics,
                "source_pdf": os.path.basename(pdf_path),
                "page": page_num + 1
            })
    doc.close()
    return questions_in_pdf

def main():
    """Main function to run the OCR processing pipeline."""
    print("🚀 Starting the OCR question processing engine...")
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
    
    pdf_files = glob.glob(os.path.join(PDF_SOURCE_DIR, "*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in '{PDF_SOURCE_DIR}'.")
        return

    print(f"🔍 Found {len(pdf_files)} PDF files to process.")
    
    final_dataset = []
    for pdf_path in pdf_files:
        try:
            final_dataset.extend(process_pdf_with_ocr(pdf_path))
        except Exception as e:
            print(f"   - ⚠️  Could not process {os.path.basename(pdf_path)}. Error: {e}")

    with open(JSON_PATH, "w") as f:
        json.dump(final_dataset, f, indent=2)
        
    print(f"\n✅ OCR processing complete! {len(final_dataset)} questions extracted.")
    print(f"   Database saved to: {JSON_PATH}")
    print(f"   Images saved in: {IMG_OUTPUT_DIR}")

if __name__ == "__main__":
    main()