import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from urlextract import URLExtract

from dotenv import load_dotenv

from predictor import PhishingPredictor

load_dotenv()

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

_extractor = URLExtract()
STATS_FILE = Path("stats.json")


def _find_urls(text: str) -> list[str]:
    return _extractor.find_urls(text, only_unique=True)

_predictor: PhishingPredictor | None = None


def _load_stats() -> dict:
    if STATS_FILE.exists():
        with open(STATS_FILE, encoding="utf-8") as f:
            data = json.load(f)
            data["users"] = set(data.get("users", []))
            return data
    return {
        "total_checks": 0,
        "phishing_detected": 0,
        "legitimate": 0,
        "users": set(),
        "last_update": "",
    }


def _save_stats(stats: dict) -> None:
    data = {**stats, "users": list(stats["users"]), "last_update": datetime.now().isoformat()}
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "👋 Привет! Я бот для проверки ссылок на фишинг.\n\n"
        "Просто отправь мне URL — я скажу, безопасный он или нет.\n\n"
        "Используется модель машинного обучения (логистическая регрессия) "
        "с точностью 96.2%.\n\n"
        "Команды:\n"
        "/help — справка\n"
        "/stats — статистика проверок"
    )
    await update.message.reply_text(text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📋 Инструкция:\n\n"
        "1. Отправь одну или несколько ссылок в одном сообщении\n"
        "2. Я проверю каждую и дам вердикт\n\n"
        "Примеры:\n"
        "• https://google.com\n"
        "• http://paypal-verify.tk/login\n\n"
        "Бот не хранит ссылки и не передаёт их третьим лицам."
    )
    await update.message.reply_text(text)


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stats = _load_stats()
    text = (
        f"📊 Статистика проверок:\n\n"
        f"Всего проверено: {stats['total_checks']}\n"
        f"Фишинговых: {stats['phishing_detected']}\n"
        f"Легитимных: {stats['legitimate']}\n"
        f"Уникальных пользователей: {len(stats['users'])}"
    )
    if stats.get("last_update"):
        text += f"\nПоследнее обновление: {stats['last_update'][:19]}"
    await update.message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    urls = _find_urls(text)

    if not urls:
        await update.message.reply_text(
            "Ссылок не найдено. Отправь URL, начинающийся с http://, https:// или www."
        )
        return

    stats = _load_stats()
    stats["users"].add(update.effective_user.id)

    for url in urls:
        result = _predictor.predict(url)
        stats["total_checks"] += 1

        if result["is_phishing"]:
            stats["phishing_detected"] += 1
            features_text = (
                "\n".join(f"• {f}" for f in result["suspicious_features"])
                if result["suspicious_features"] else "—"
            )
            reply = (
                f"⚠️ ФИШИНГ (опасно!)\n"
                f"{result['url']}\n\n"
                f"Вероятность фишинга: {result['probability'] * 100:.1f}%\n\n"
                f"Подозрительные признаки:\n{features_text}\n\n"
                f"Рекомендация: НЕ ПЕРЕХОДИТЕ по этой ссылке."
            )
        else:
            stats["legitimate"] += 1
            reply = (
                f"✅ ЛЕГИТИМНЫЙ\n"
                f"{result['url']}\n\n"
                f"Вероятность фишинга: {result['probability'] * 100:.1f}%\n\n"
                f"Подозрительных признаков не обнаружено."
            )

        await update.message.reply_text(reply)
        logger.info("Проверен URL: %s → %s (%.3f)", url, result["label"], result["probability"])

    _save_stats(stats)


def main() -> None:
    global _predictor

    token = os.environ.get("BOT_TOKEN")
    if not token:
        logger.error("Переменная BOT_TOKEN не задана. Завершение.")
        sys.exit(1)

    try:
        _predictor = PhishingPredictor()
        logger.info("Модель загружена успешно.")
    except FileNotFoundError as e:
        logger.error("Не найден pkl-файл: %s. Сначала запустите phishing_classifier.py", e)
        sys.exit(1)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()
