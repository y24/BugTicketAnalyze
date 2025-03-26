# 必要な再インポート
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import learning_curve

# 分類レポート
def display_classification_report(y_test, y_pred, target_labels):
    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_labels))

# 混同行列の可視化
def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("混同行列（Confusion Matrix）")
    plt.xlabel("予測ラベル")
    plt.ylabel("実際のラベル")
    plt.show()

# 学習曲線の描画
def plot_learning_curve(model, X, y, scoring='f1_weighted', cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label="訓練スコア")
    plt.plot(train_sizes, test_mean, label="検証スコア")
    plt.xlabel("訓練データのサイズ")
    plt.ylabel(f"{scoring}")
    plt.title("学習曲線（Learning Curve）")
    plt.legend()
    plt.grid(True)
    plt.show()

# 特徴量の重要度（ランダムフォレスト + TF-IDF の場合）
def show_feature_importance(model, vectorizer, top_n=10):
    try:
        importances = model.feature_importances_
        feature_names = vectorizer.get_feature_names_out()
        top_indices = importances.argsort()[-top_n:][::-1]
        print(f"📌 上位 {top_n} の重要特徴量:")
        for i in top_indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
    except AttributeError:
        print("このモデルでは特徴量の重要度を取得できません。")
