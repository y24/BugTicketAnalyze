import re
import MeCab

# 形態素解析を用いた前処理
def preprocess_text(text, ignores: list[str]=[]):
    text = text.lower()  # 小文字化
    text = re.sub(r'\W+', ' ', text)  # 特殊文字削除
    mecab = MeCab.Tagger("-Owakati")  # 分かち書き
    tokens = mecab.parse(text).strip().split()

    # 除外ワードを取り除く
    filtered_tokens = [t for t in tokens if t not in ignores]
    return " ".join(filtered_tokens)