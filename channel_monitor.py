import asyncio
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from telethon import TelegramClient, events
from urlextract import URLExtract

from dotenv import load_dotenv

from predictor import PhishingPredictor

load_dotenv()

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("monitor_errors.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

_extractor = URLExtract()


def _find_urls(text: str) -> list[str]:
    return _extractor.find_urls(text, only_unique=True)


CSV_FILE = Path("monitoring_log.csv")
CSV_HEADERS = ["timestamp", "channel", "message_id", "url", "is_phishing", "probability", "features"]


def _log_csv(row: dict) -> None:
    write_header = not CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


async def _notify_saved(client: TelegramClient, text: str) -> None:
    try:
        await client.send_message("me", text)
    except Exception as e:
        logger.error("Не удалось отправить уведомление в Избранное: %s", e)


async def main() -> None:
    api_id_str = os.environ.get("TG_API_ID")
    api_hash = os.environ.get("TG_API_HASH")
    phone = os.environ.get("TG_PHONE")
    channels_raw = os.environ.get("MONITOR_CHANNELS", "")

    missing = [k for k, v in {
        "TG_API_ID": api_id_str,
        "TG_API_HASH": api_hash,
        "TG_PHONE": phone,
        "MONITOR_CHANNELS": channels_raw,
    }.items() if not v]

    if missing:
        logger.error("Не заданы переменные окружения: %s", ", ".join(missing))
        sys.exit(1)

    api_id = int(api_id_str)
    channels = [c.strip() for c in channels_raw.split(",") if c.strip()]

    try:
        predictor = PhishingPredictor()
        logger.info("Модель загружена успешно.")
    except FileNotFoundError as e:
        logger.error("Не найден pkl-файл: %s. Сначала запустите phishing_classifier.py", e)
        sys.exit(1)

    client = TelegramClient("session_monitor", api_id, api_hash)

    @client.on(events.NewMessage(chats=channels))
    async def on_message(event: events.NewMessage.Event) -> None:
        text = event.message.text or ""
        urls = _find_urls(text)
        if not urls:
            return

        channel_name = getattr(event.chat, "username", None) or str(event.chat_id)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for url in urls:
            result = predictor.predict(url)
            logger.info("[%s] %s → %s (%.3f)", channel_name, url, result["label"], result["probability"])

            _log_csv({
                "timestamp": timestamp,
                "channel": channel_name,
                "message_id": event.message.id,
                "url": url,
                "is_phishing": result["is_phishing"],
                "probability": round(result["probability"], 4),
                "features": "; ".join(result["suspicious_features"]),
            })

            if result["is_phishing"]:
                features_text = ", ".join(result["suspicious_features"]) or "—"
                notification = (
                    f"⚠️ ФИШИНГ ОБНАРУЖЕН\n\n"
                    f"Канал: @{channel_name}\n"
                    f"Время: {timestamp}\n"
                    f"URL: {url}\n"
                    f"Вероятность: {result['probability'] * 100:.1f}%\n\n"
                    f"Признаки: {features_text}"
                )
                await _notify_saved(client, notification)

    logger.info("Подключение к Telegram...")
    await client.start(phone=phone)
    logger.info("Мониторинг каналов: %s", channels)

    while True:
        try:
            await client.run_until_disconnected()
            break
        except Exception as e:
            logger.error("Соединение разорвано: %s. Переподключение через 10 сек...", e)
            await asyncio.sleep(10)
            try:
                await client.connect()
            except Exception as e2:
                logger.error("Переподключение не удалось: %s", e2)
                break


if __name__ == "__main__":
    asyncio.run(main())
