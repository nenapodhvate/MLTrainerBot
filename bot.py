import os
import pandas as pd
import io
import logging
from telegram import Update, Message
from telegram.ext import Application, CommandHandler, MessageHandler, filters, \
    ContextTypes
from dotenv import load_dotenv
from model import ModelTrainer 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне CSV-файл с данными для обучения модели.")


async def handle_csv_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await context.bot.get_file(update.message.document.file_id)
    file_path = io.BytesIO(await file.download_as_bytearray())

    try:
        data = pd.read_csv(file_path)

        if data.empty:
            logger.error("Загруженный CSV файл пуст.")
            await update.message.reply_text(
                "Ошибка: загруженный файл пуст. Пожалуйста, загрузите файл с данными.")
            return

        logger.info("Первые 5 строк данных:")
        logger.info(data.head())

        model_trainer = ModelTrainer()

        accuracy = await model_trainer.train_model(data, context.bot,
                                                   update.message.chat.id)

        if accuracy is not None:
            await update.message.reply_text(
                f"Обучение завершено с точностью: {accuracy:.2f}")
        else:
            await update.message.reply_text(
                "Произошла ошибка при обучении модели.")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла CSV: {e}")
        await update.message.reply_text(f"Ошибка при обработке данных: {e}")


async def handle_unexpected_message(update: Update,
                                    context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ошибка: я принимаю только CSV-файлы. Пожалуйста, отправьте файл в правильном формате.")


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(filters.Document.FileExtension("csv"), handle_csv_file))

    app.add_handler(
        MessageHandler(filters.ALL & ~filters.Document.FileExtension("csv"),
                       handle_unexpected_message))

    app.run_polling()


if __name__ == "__main__":
    main()