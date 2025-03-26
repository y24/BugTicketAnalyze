# -------------------
# 1. データの読み込みと前処理
# -------------------
import pandas as pd
from preprocess import preprocess_text

# CSVデータの読み込み
df = pd.read_csv("bug_tickets_3class.csv")

# テキストを前処理
df["preprocessed_text"] = df["詳細"].apply(preprocess_text)
# print(df[["詳細", "preprocessed_text"]])

# -------------------
# 2. 特徴量エンジニアリング
# -------------------

# (1) TF-IDF による特徴量変換
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF ベクトル化
vectorizer = TfidfVectorizer(max_features=5000)  # 最大5000単語まで考慮
X_tfidf = vectorizer.fit_transform(df["preprocessed_text"])

# TF-IDFの特徴語リストを取得
print("\n特徴語:", vectorizer.get_feature_names_out())


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
# 3. クラスタリングによる傾向分類
# -------------------

# (1) K-Means を用いたクラスタリング
from sklearn.cluster import KMeans

num_clusters = 3  # クラスター数
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(X_tfidf)

# クラスタごとのデータを確認
print("\n", df[["タイトル", "cluster"]])


# (2) PCA による可視化
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="viridis")
plt.colorbar()
plt.show()

# -------------------
# 4. 重要度の数値化
# -------------------

# (1) 重要度を教師あり学習で分類
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ラベルを数値化
df["重要度"] = df["重要度"].map({"低": 0, "中": 1, "高": 2})

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df["重要度"], test_size=0.2, random_state=42)

# モデル学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 精度評価
y_pred = model.predict(X_test)
# 結果を表示
print("\n", classification_report(y_test, y_pred, target_names=["低", "中", "高"]))


# (2) 重要度スコアの算出
import numpy as np

importance_scores = model.predict_proba(X_tfidf)
df["importance_score"] = np.max(importance_scores, axis=1)


# 5. モデルの保存
import joblib

joblib.dump(model, "models/3class_bug_severity_model.pkl")
joblib.dump(vectorizer, "models/3class_tfidf_vectorizer.pkl")
