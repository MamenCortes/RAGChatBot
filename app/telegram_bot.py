import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from huggingface_hub import InferenceClient

from .config import settings
from .rag import rag_answer

user_conversations: dict[int, list[dict]] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Ask me something about the documents.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    text = update.message.text.strip()

    hf_client: InferenceClient = context.application.bot_data["hf_client"]
    await update.message.reply_text("Thinking... 🤔")

    history = user_conversations.setdefault(user_id, [])
    history.append({"role": "user", "content": text})

    try:
        answer = rag_answer(
            hf_client=hf_client,
            user_message=text,
            chat_history=history,
            top_k=settings.top_k,
            model=settings.llm_model_name,
        )
        history.append({"role": "assistant", "content": answer})
        await update.message.reply_text(answer)
    except Exception as e:
        print(f"Error: {e}")
        await update.message.reply_text("Something went wrong! 😥")

def main():
    load_dotenv(".env")

    hf_client = InferenceClient(
        provider="novita",
        api_key=settings.hf_api_key,
    )

    app = ApplicationBuilder().token(settings.telegram_token).build()
    app.bot_data["hf_client"] = hf_client

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
