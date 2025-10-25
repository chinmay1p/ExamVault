import json
import os

JSON_PATH = os.path.join("question_bank_ocr", "data.json")

def load_question_bank():
    with open(JSON_PATH, "r") as f:
        return json.load(f)

def search_by_topic(topic, db):
    topic_lower = topic.lower()
    results = []
    for question in db:
        if any(topic_lower in t.lower() for t in question["topics"]):
            results.append(question)
    return results

def display_results(results, topic):
    print("\n" + "="*50)
    if not results:
        print(f"No questions found for the topic: '{topic}'")
    else:
        print(f" Found {len(results)} questions for '{topic}':\n")
        for i, res in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"Source : {res['source_pdf']} (Year: {res['year']}, Page: {res['page']})")
            print(f"Topics : {', '.join(res['topics'])}")
            print(f"Image  : {res['image_path']}")
            print("\n" + res['question_text'])
            print("-" * 15 + "\n")
    print("="*50 + "\n")
    
def main():
    print("🎓 Welcome to the Question Bank Search App!")
    db = load_question_bank()
    
    if not db:
        return 
        
    while True:
        topic = input("Enter a topic to search (e.g., 'SVM', 'Clustering') or type 'q' to quit: ")
        if topic.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        results = search_by_topic(topic, db)
        display_results(results, topic)
        
if __name__ == "__main__":
    main()