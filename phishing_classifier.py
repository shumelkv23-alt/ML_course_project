import pickle
import re
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")

DATASET_PATH = "phishing_site_urls.csv"

print("=" * 60)
print("1. ЗАГРУЗКА ДАТАСЕТА")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"Исходный размер: {len(df)} записей")

df = df.drop_duplicates(subset=["URL"]).dropna()
df["URL"] = df["URL"].astype(str).str.lower()

if not pd.api.types.is_numeric_dtype(df["Label"]):
    df["Label"] = df["Label"].map({"good": 0, "bad": 1})

df = df.dropna(subset=["Label"])
df["Label"] = df["Label"].astype(int)

print(f"После очистки: {len(df)} записей")
print(f"Легитимных (0): {(df['Label'] == 0).sum()}")
print(f"Фишинговых (1): {(df['Label'] == 1).sum()}")

plt.figure(figsize=(6, 4))
df["Label"].value_counts().plot(
    kind="bar", color=["#2E86AB", "#E63946"]
)
plt.title("Распределение классов в датасете")
plt.xlabel("Класс (0 - легитимный, 1 - фишинговый)")
plt.ylabel("Количество записей")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()
print("Сохранён график: class_distribution.png")


print("\n" + "=" * 60)
print("2. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ")
print("=" * 60)

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account",
    "update", "password", "banking", "paypal", "bank",
]

IP_REGEX = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def extract_features(url: str) -> list:
    if "://" not in url:
        url_full = "http://" + url
    else:
        url_full = url

    try:
        parsed = urlparse(url_full)
        domain = parsed.netloc or url
    except ValueError:
        domain = url

    return [
        len(url),
        len(domain),
        url.count("."),
        url.count("-"),
        url.count("/"),
        1 if IP_REGEX.match(domain) else 0,
        1 if "@" in url else 0,
        1 if url.startswith("https") else 0,
        domain.count("."),
        int(any(w in url for w in SUSPICIOUS_WORDS)),
        1 if len(url) > 75 else 0,
    ]


FEATURE_NAMES = [
    "url_length", "domain_length", "dot_count", "dash_count",
    "slash_count", "is_ip", "has_at", "is_https",
    "subdomain_count", "has_suspicious_word", "is_long",
]

print("Извлекаю ручные признаки...")
manual_features = np.array([extract_features(u) for u in df["URL"]])
print(f"Размерность ручных признаков: {manual_features.shape}")

scaler = StandardScaler()
manual_scaled = scaler.fit_transform(manual_features)

print("Вычисляю TF-IDF...")
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=5000,
    lowercase=True,
)
X_tfidf = tfidf.fit_transform(df["URL"])
print(f"Размерность TF-IDF: {X_tfidf.shape}")

X = hstack([csr_matrix(manual_scaled), X_tfidf])
y = df["Label"].values
print(f"Итоговая размерность: {X.shape}")


print("\n" + "=" * 60)
print("3. РАЗБИВКА TRAIN/TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Обучающая выборка: {X_train.shape[0]} записей")
print(f"Тестовая выборка:  {X_test.shape[0]} записей")


print("\n" + "=" * 60)
print("4. ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
print("=" * 60)

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
)

print("Обучение...")
model.fit(X_train, y_train)
print("Модель обучена.")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(FEATURE_NAMES, f)
print("Модель и препроцессинг сохранены в pkl-файлы.")


print("\n" + "=" * 60)
print("5. ОЦЕНКА КАЧЕСТВА НА ТЕСТЕ")
print("=" * 60)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nОтчёт классификации:")
print(classification_report(
    y_test, y_pred, target_names=["Легитимный", "Фишинговый"]
))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC:      {roc_auc_score(y_test, y_proba):.4f}")


print("\n" + "=" * 60)
print("6. ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Легитимный", "Фишинговый"],
    yticklabels=["Легитимный", "Фишинговый"],
)
plt.xlabel("Предсказание модели")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок логистической регрессии")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Сохранена: confusion_matrix.png")


fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#E63946", lw=2,
         label=f"Логистическая регрессия (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()
print("Сохранена: roc_curve.png")


print("\n" + "=" * 60)
print("7. ПОДБОР ПАРАМЕТРА РЕГУЛЯРИЗАЦИИ C")
print("=" * 60)

sample_size = min(50000, X_train.shape[0])
idx = np.random.RandomState(42).choice(X_train.shape[0], sample_size, replace=False)
X_sample = X_train[idx]
y_sample = y_train[idx]

c_values = [0.01, 0.1, 1.0, 10.0]
cv_scores = []

for c in c_values:
    m = LogisticRegression(C=c, max_iter=500, class_weight="balanced",
                            solver="lbfgs", random_state=42)
    scores = cross_val_score(m, X_sample, y_sample, cv=3,
                              scoring="f1", n_jobs=-1)
    cv_scores.append(scores.mean())
    print(f"  C = {c:>6}: F1 = {scores.mean():.4f}")

plt.figure(figsize=(7, 4))
plt.plot(c_values, cv_scores, marker="o", color="#2E86AB", lw=2)
plt.xscale("log")
plt.xlabel("Параметр регуляризации C")
plt.ylabel("F1-score (кросс-валидация)")
plt.title("Подбор оптимального C")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("c_tuning.png", dpi=150)
plt.close()
print("Сохранена: c_tuning.png")


print("\n" + "=" * 60)
print("8. СРАВНЕНИЕ С НАИВНЫМ БАЙЕСОМ")
print("=" * 60)

nb_model = MultinomialNB()
nb_model.fit(X_tfidf[idx][:int(0.8 * sample_size)],
             y[idx][:int(0.8 * sample_size)])
nb_pred = nb_model.predict(X_tfidf[idx][int(0.8 * sample_size):])
y_nb_true = y[idx][int(0.8 * sample_size):]

print(f"Логистическая регрессия: Acc={accuracy_score(y_test, y_pred):.4f}  "
      f"F1={f1_score(y_test, y_pred):.4f}")
print(f"Наивный байес:           Acc={accuracy_score(y_nb_true, nb_pred):.4f}  "
      f"F1={f1_score(y_nb_true, nb_pred):.4f}")


print("\n" + "=" * 60)
print("9. ДЕМО — ПРОВЕРКА КОНКРЕТНЫХ URL")
print("=" * 60)

test_urls = [
    "https://www.google.com/search?q=python",
    "https://github.com/scikit-learn",
    "http://paypal-secure-login.verify-account.tk/",
    "http://192.168.1.45/bank/login.php",
    "http://telegram-premium-free.gift-claim.cf/login",
]


def predict_url(url: str):
    manual = scaler.transform([extract_features(url.lower())])
    tfidf_vec = tfidf.transform([url.lower()])
    x = hstack([csr_matrix(manual), tfidf_vec])
    proba = model.predict_proba(x)[0, 1]
    label = "ФИШИНГ" if proba > 0.5 else "легитимный"
    return label, proba


for url in test_urls:
    label, proba = predict_url(url)
    print(f"  [{label:>10}]  (p={proba:.3f})  {url}")


print("\n" + "=" * 60)
print("ВСЁ ГОТОВО.")
print("Файлы графиков сохранены в текущей директории:")
print("  - class_distribution.png")
print("  - confusion_matrix.png")
print("  - roc_curve.png")
print("  - c_tuning.png")
print("=" * 60)
