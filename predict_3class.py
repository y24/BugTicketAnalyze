# -------------------
# 6. 利用サンプル
# -------------------

import joblib
from preprocess import preprocess_text

# モデルとベクトライザをロード
model = joblib.load("models/3class_bug_severity_model.pkl")
vectorizer = joblib.load("models/3class_tfidf_vectorizer.pkl")

# 新規データの分類
new_text = "スワードリセットのメールがユーザーに届きません"
new_text_preprocessed = preprocess_text(new_text)
new_vector = vectorizer.transform([new_text_preprocessed])

# 数値で予測
pred_num = model.predict(new_vector)[0]
print("予測された重要度（数値）:", pred_num)

# ラベルに変換
severity_map = {"高": 2, "中": 1, "低": 0}
reverse_map = {v: k for k, v in severity_map.items()}
pred_label = reverse_map[pred_num]
print("予測された重要度（ラベル）:", pred_label)