import joblib
from pathlib import Path
import sys

# Set standard output encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

vectorizer_path = Path("models/sentiment/tfidf_vectorizer.pkl")
model_path = Path("models/sentiment/rf_classifier.pkl")

if vectorizer_path.exists() and model_path.exists():
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    
    test_sentences = [
        "máy dùng rất tốt pin trâu mượt mà",
        "máy quá tệ lag giật đơ màn hình nóng máy",
        "nhân viên tư vấn nhiệt tình thái độ tốt",
        "giao hàng chậm trễ chăm sóc khách hàng kém",
        "pin cùi bắp tụt nhanh sạc nóng"
    ]
    
    for s in test_sentences:
        X = vectorizer.transform([s])
        prob = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        print(f"Sentence: '{s}'")
        print(f"  Classes: {list(model.classes_)}")
        print(f"  Probabilities: {list(prob)}")
        print(f"  Prediction: {pred}")
else:
    print("Models do not exist.")
