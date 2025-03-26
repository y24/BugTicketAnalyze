import json 
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

model_dir = config["model"]["dir"]
model_name = config["model"]["name"]
vectorizer_name = config["model"]["vectorizer_name"]
severity_map = config["severity_map"]

# -------------------
# 6. 利用サンプル
# -------------------

import joblib
from preprocess import preprocess_text

# モデルとベクトライザをロード
model = joblib.load(f"{model_dir}/{model_name}")
vectorizer = joblib.load(f"{model_dir}/{vectorizer_name}")

# 新規データの分類
new_text = "スワードリセットのメールがユーザーに届きません"
new_text_preprocessed = preprocess_text(new_text)
new_vector = vectorizer.transform([new_text_preprocessed])

# 数値で予測
pred_num = model.predict(new_vector)[0]
print("予測された重要度（数値）:", pred_num)

# ラベルに変換
reverse_map = {v: k for k, v in severity_map.items()}
pred_label = reverse_map[pred_num]
print("予測された重要度（ラベル）:", pred_label)