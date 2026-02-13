# To import the tokens
import os
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from huggingface_hub import InferenceClient

TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
HF_API_KEY = 'YOUR_HF_API_TOKEN'

user_conversations = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Send me a message and I will reply using AI!')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    user_message = update.message.text
    # Get the HF client created in main()
    client: InferenceClient = context.application.bot_data["hf_client"]
    await update.message.reply_text('Thinking... 🤔')

    if user_id not in user_conversations:
        user_conversations[user_id] = []

    user_conversations[user_id].append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=user_conversations[user_id],
            max_tokens=500,
        )

        response_text = completion.choices[0].message["content"]
        user_conversations[user_id].append({"role": "assistant", "content": response_text})

        await update.message.reply_text(response_text)

    except Exception as e:
        print(f"Error: {e}")
        await update.message.reply_text('Something went wrong! 😥')

def main():
    load_dotenv(".env")

    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
    HF_API_KEY = os.environ["HF_API_KEY"]

    hf_client = InferenceClient(
        provider="novita",
        api_key=HF_API_KEY,
    )

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Make hf_client available inside handlers via context.application.bot_data
    app.bot_data["hf_client"] = hf_client

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()