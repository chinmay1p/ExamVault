import json
import pandas as pd
import joblib
import numpy as np

# ML Imports
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

# --- Configuration ---
JSON_PATH = "question_bank_ocr/data.json"
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Output File Paths ---
XGB_MODEL_PATH = 'xgb_multi_output_model.joblib' 
MLB_PATH = 'multi_label_binarizer.joblib'


def load_data(json_path):
    """Loads the JSON dataset and converts it to a pandas DataFrame."""
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f" ERROR: Data file not found at {json_path}")
        print("Please run the processor.py script first to generate the data.")
        return None
    except json.JSONDecodeError:
        print(f" ERROR: Could not decode JSON from {json_path}. File might be empty or corrupt.")
        return None

    if not data:
        print(" ERROR: No data found in the JSON file.")
        return None
        
    df = pd.DataFrame(data)
    print(f" Loaded {len(df)} questions.")
    return df

def prepare_labels(df):
    """Fits and transforms the topic tags using MultiLabelBinarizer."""
    print("Preparing and binarizing labels (Y)...")
    topics = df['topics'].tolist()
    
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(topics)
    
    print(f"Found {len(mlb.classes_)} unique tags.")
    print(f"Saving binarizer to {MLB_PATH}...")
    joblib.dump(mlb, MLB_PATH)
    print("✅ MultiLabelBinarizer saved.")
    
    return y_binary, mlb

def prepare_features(df):
    """Creates text embeddings using a SentenceTransformer model."""
    print(f"Loading SentenceTransformer model: '{SBERT_MODEL_NAME}'...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    
    print("Creating text embeddings (X)... This may take a few moments.")
    texts = df['question_text'].tolist()
    X_embeddings = sbert_model.encode(texts, show_progress_bar=True)
    
    print("✅ Text embeddings created.")
    return X_embeddings

def train_classifier(X_train, y_train):
    """Initializes and trains the MultiOutput XGBoost classifier."""
    print("Training XGBoost model...")
    
    # Define the base classifier (XGBoost)
    # These parameters are a good starting point for binary classification
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        base_score=0.5,
        use_label_encoder=False,
        random_state=42
    )
    
    model = MultiOutputClassifier(xgb, n_jobs=-1)
        
    model.fit(X_train, y_train)
    
    print("✅ Model training complete.")
    print(f"Saving trained model to {XGB_MODEL_PATH}...")
    joblib.dump(model, XGB_MODEL_PATH)
    print("✅ Trained XGBoost model saved.")
    
    return model

def evaluate_model(model, X_test, y_test, mlb):
    """Evaluates the model on the test set and prints a report."""
    print("\n--- Model Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    
    # Subset Accuracy: Measures the percentage of samples where all labels were predicted correctly.
    # This is a very strict metric and is often low in multi-label problems.
    subset_accuracy = accuracy_score(y_test, y_pred)
    print(f"Subset Accuracy (Exact Match): {subset_accuracy:.4f}")
    
    # Hamming Loss: The fraction of labels that are incorrectly predicted.
    # (Lower is better)
    hamming = hamming_loss(y_test, y_pred)
    print(f"Hamming Loss (Avg. Error): {hamming:.4f}")
    
    # Classification Report: Shows precision, recall, and F1-score for each individual tag.
    # This is the most useful metric for understanding per-tag performance.
    print("\nClassification Report (per-tag):")
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=mlb.classes_,
        zero_division=0
    )
    print(report)

def main():
    """Main function to run the full training pipeline."""
    
    # 1. Load Data
    df = load_data(JSON_PATH)
    if df is None:
        return

    # 2. Prepare Y (Labels)
    y_binary, mlb = prepare_labels(df)
    
    # 3. Prepare X (Features)
    X_embeddings = prepare_features(df)
    
    # 4. Split Data
    print("Splitting data into training and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, 
        y_binary, 
        test_size=0.2, 
        random_state=42
    )
    
    # 5. Train Model
    model = train_classifier(X_train, y_train)
    
    # 6. Evaluate Model
    evaluate_model(model, X_test, y_test, mlb)
    
    print("\n🚀 Training pipeline complete!")
    print(f"You can now run 'processor.py', which will use the saved models:")
    print(f"  - {MLB_PATH}")
    print(f"  - {XGB_MODEL_PATH}")

if __name__ == "__main__":
    main()