import re
import MeCab

# 形態素解析を用いた前処理
def preprocess_text(text):
    text = text.lower()  # 小文字化
    text = re.sub(r'\W+', ' ', text)  # 特殊文字削除
    mecab = MeCab.Tagger("-Owakati")  # 分かち書き
    return mecab.parse(text).strip()