import json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# データディレクトリ
data_dir = config["csv"]["dir"]
# データCSV
data_csv = config["csv"]["data"]
# 無視する単語CSV
ignores_csv = config["csv"]["ignores"]
# 学習対象カラム
source_column = config["columns"]["source"]
# 予測対象カラム
target_column = config["columns"]["target"]
# 重要度の数値化
severity_map = config["severity_map"]
# モデルの保存先
model_dir = config["model"]["dir"]
model_name = config["model"]["name"]
vectorizer_name = config["model"]["vectorizer_name"]

# -------------------
# 1. データの読み込みと前処理
# -------------------
import pandas as pd
from preprocess import preprocess_text

# データ読み込み
df = pd.read_csv(f"{data_dir}/{data_csv}")

# 無視する単語の読み込み
ignore_df = pd.read_csv(f"{data_dir}/{ignores_csv}")
ignores = ignore_df["ignore"].tolist()

# テキストを前処理
df["preprocessed_text"] = df[source_column].apply(preprocess_text, ignores=ignores)
print(df[[source_column, "preprocessed_text"]], "\n")

# -------------------
# 2. 特徴量エンジニアリング
# -------------------

# (1) TF-IDF による特徴量変換
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF ベクトル化
vectorizer = TfidfVectorizer(max_features=5000)  # 最大5000単語まで考慮
X_tfidf = vectorizer.fit_transform(df["preprocessed_text"])

# TF-IDFの特徴語リストを取得
print("特徴語:", vectorizer.get_feature_names_out(), "\n")


# (2) Word2Vec による単語埋め込み
from gensim.models import Word2Vec
import numpy as np

# 形態素解析済みのデータをリストに変換
tokenized_texts = [text.split() for text in df["preprocessed_text"]]

# Word2Vec モデルの学習
model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# 各文章の単語ベクトルの平均を求める
X_word2vec = np.array([np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(100)], axis=0)
                       for words in tokenized_texts])

# -------------------
# 4. 重要度の数値化
# -------------------

# (1) 重要度を教師あり学習で分類
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ラベルを数値化
df["target_num"] = df[target_column].map(severity_map)

# 特徴量（TF-IDFなど）とラベルの用意
X = X_tfidf  # または X_word2vec
y = df["target_num"]

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 精度評価
y_pred = model.predict(X_test)

# 学習結果を表示
import visualize
# 精度レポート表示
visualize.display_classification_report(y_test, y_pred, target_labels=severity_map.keys())

# 混同行列の可視化
# learning_report.plot_confusion_matrix(y_test, y_pred, labels=labels)

# 学習曲線（Xとyは全データ）
# learning_report.plot_learning_curve(model, X, y, scoring='f1_weighted')

# 重要単語（TF-IDF + ランダムフォレストの場合）
visualize.show_feature_importance(model, vectorizer, top_n=20)


# (2) 重要度スコアの算出
import numpy as np

importance_scores = model.predict_proba(X_tfidf)
df["importance_score"] = np.max(importance_scores, axis=1)


# 5. モデルの保存
import joblib

joblib.dump(model, f"{model_dir}/{model_name}")
joblib.dump(vectorizer, f"{model_dir}/{vectorizer_name}")
