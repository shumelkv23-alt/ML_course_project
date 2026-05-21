# Классификатор фишинговых URL для Telegram

Курсовой проект по дисциплине «Машинное обучение и нейронные сети».
Тема: классификация фишинговых URL-адресов из Telegram-каналов с использованием логистической регрессии.

## Возможности

- **Обучение модели** на датасете легитимных и фишинговых URL (~96% accuracy, AUC ≈ 0.99).
- **Telegram-бот** для интерактивной проверки ссылок пользователями.
- **Мониторинг каналов** через Telethon — автоматическое обнаружение фишинга в потоке сообщений.
- **Извлечение признаков**: ручные (длина, дефисы, TLD, IP, ключевые слова) + TF-IDF на символьных n-граммах.

## Структура проекта

```
.
├── phishing_classifier.py   # обучение модели и генерация графиков
├── predictor.py             # общий класс PhishingPredictor
├── telegram_bot.py          # интерактивный бот (python-telegram-bot)
├── channel_monitor.py       # мониторинг каналов (telethon)
├── requirements.txt
├── .env.example             # шаблон переменных окружения
└── LICENSE
```

## Установка

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

## Подготовка данных и обучение модели

1. Скачай датасет с Kaggle:
   [phishing-site-urls](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
2. Положи файл `phishing_site_urls.csv` в корень проекта.
3. Запусти обучение:

```bash
python phishing_classifier.py
```

После обучения появятся файлы модели (`model.pkl`, `scaler.pkl`, `tfidf.pkl`, `feature_names.pkl`)
и графики (`class_distribution.png`, `confusion_matrix.png`, `roc_curve.png`, `c_tuning.png`).

## Настройка окружения

Скопируй шаблон и заполни значениями:

```bash
cp .env.example .env
```

Переменные:

| Ключ               | Где взять                                          |
|--------------------|----------------------------------------------------|
| `BOT_TOKEN`        | [@BotFather](https://t.me/BotFather) в Telegram    |
| `TG_API_ID`        | https://my.telegram.org → API development tools    |
| `TG_API_HASH`      | там же                                             |
| `TG_PHONE`         | номер аккаунта Telegram (с кодом страны)           |
| `MONITOR_CHANNELS` | список каналов через запятую (`channel1,channel2`) |

## Запуск

**Telegram-бот** (требуется `BOT_TOKEN`):

```bash
python telegram_bot.py
```

**Мониторинг каналов** (требуется `TG_API_ID`, `TG_API_HASH`, `TG_PHONE`, `MONITOR_CHANNELS`):

```bash
python channel_monitor.py
```

При первом запуске Telethon попросит код подтверждения из Telegram — это нормально.
Уведомления о найденном фишинге отправляются в «Избранное» (Saved Messages).

## Используемые технологии

- **Python 3.10+**
- **scikit-learn** — логистическая регрессия, TF-IDF, кросс-валидация
- **pandas / numpy / scipy** — обработка данных
- **matplotlib / seaborn** — графики
- **python-telegram-bot** — интерактивный бот
- **telethon** — клиент Telegram MTProto для мониторинга
- **urlextract** — извлечение URL по Public Suffix List
