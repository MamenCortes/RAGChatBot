import os
import csv
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, UTC
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from huggingface_hub import InferenceClient

from .config import settings
from .rag import rag_answer, rag_answer_all_modes

datetime.now(UTC).isoformat()

user_conversations: dict[int, list[dict]] = {}

# Eval state per user: stores the last TripleAnswer while waiting for a vote
user_eval_state: dict[int, dict] = {}
# Set of user_ids currently in eval mode
eval_mode_users: set[int] = set()

EVAL_LOG_PATH = Path("eval/eval_results.csv")
MODES = ["semantic", "hybrid", "language_aware_hybrid", "no_retrieval"]
LABELS = {
    "semantic":     "Semantic",
    "hybrid":       "Hybrid (Semantic + Keyword)",
    "language_aware_hybrid":    "Language-aware Hybrid",
    "no_retrieval": "No retrieval (LLM only)",
}

# ── CSV helpers ────────────────────────────────────────────────────────────────

def _ensure_csv() -> None:
    """Ensure the CSV log file exists and has the correct header."""
    if not EVAL_LOG_PATH.exists():
        with open(EVAL_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_id", "query", "preferred_mode",
                             "answer_semantic", "answer_hybrid", "answer_language_aware_hybrid", "answer_no_retrieval"])


def _log_eval(user_id: int, query: str, preferred_mode: str, triple) -> None:
    """Append an evaluation result to the CSV log."""
    _ensure_csv()
    with open(EVAL_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(UTC).isoformat(),
            user_id,
            query,
            preferred_mode,
            triple.semantic,
            triple.hybrid,
            triple.language_aware_hybrid,
            triple.no_retrieval,
        ])

# ── Command handlers ───────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the user starts the bot."""
    await update.message.reply_text(
        "Hola! Estoy aquí para contestar todas tus preguntas.\n\n"
        "Use /eval para empezar el proceso de evaluación, /stopeval para detenerlo."
    )

async def eval_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Activate evaluation mode for the user, which generates multiple answers and asks for feedback on which is best."""
    user_id = update.message.chat_id
    eval_mode_users.add(user_id)
    user_eval_state.pop(user_id, None)
    await update.message.reply_text(
    "📊 *Modo de evaluación ACTIVADO*\n\n"
    "Para cada consulta generaré varias respuestas utilizando diferentes estrategias de recuperación:\n"
    "  1️⃣  Búsqueda semántica\n"
    "  2️⃣  Búsqueda híbrida (semántica + keyword)\n"
    "  3️⃣  Búsqueda híbrida adaptada al idioma (semántica + keyword)\n"
    "  4️⃣  Sin recuperación (solo el LLM)\n\n"
    "Después de leer las respuestas, pulsa el botón de la que prefieras. "
    "Tu elección se guardará en el registro de evaluación.\n\n"
    "Envía /stopeval en cualquier momento para salir del modo de evaluación.",
    parse_mode="Markdown",
)


async def eval_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Deactivate evaluation mode for the user, returning to normal single-answer behavior."""
    user_id = update.message.chat_id
    eval_mode_users.discard(user_id)
    user_eval_state.pop(user_id, None)
    await update.message.reply_text(
    "✅ Modo de evaluación DESACTIVADO. Volviendo al modo normal de respuesta."
)

# ── Message handler ────────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages: either answer normally or generate multiple answers for evaluation."""
    user_id = update.message.chat_id
    text = update.message.text.strip()
    hf_client: InferenceClient = context.application.bot_data["hf_client"]
    history = user_conversations.setdefault(user_id, [])

    # ── Normal mode ──────────────────────────────────────────────────────────
    if user_id not in eval_mode_users:
        await update.message.reply_text("Pensando... 🤔")
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
            await update.message.reply_text("¡Algo ha salido mal! 😥")
        return

    # ── Eval mode ────────────────────────────────────────────────────────────
    await update.message.reply_text("Pensando... 🤔")
    history.append({"role": "user", "content": text})

    try:
        multiple = rag_answer_all_modes(
            hf_client=hf_client,
            user_message=text,
            chat_history=history,
            top_k=settings.top_k,
            model=settings.llm_model_name,
        )
    except Exception as e:
        print(f"Error: {e}")
        await update.message.reply_text("¡Algo ha salido mal! 😥")
        return

    # Store state so the callback can access query + triple
    user_eval_state[user_id] = {"query": text, "triple": multiple}

    # Send the three answers
    answers = {
        "semantic":     multiple.semantic,
        "hybrid":       multiple.hybrid,
        "language_aware_hybrid": multiple.language_aware_hybrid,
        "no_retrieval": multiple.no_retrieval,
    }
    for i, (mode, answer) in enumerate(answers.items(), start=1):
        await update.message.reply_text(
            f"{i}. *{LABELS[mode]}*\n\n{answer}",
        )
        #parse_mode="Markdown",

    # Inline keyboard with one button per answer
    keyboard = [[
        InlineKeyboardButton(f"{i}. {LABELS[mode]}", callback_data=f"eval_vote:{user_id}:{mode}")
        for i, mode in enumerate(MODES, start=1)
    ]]
    await update.message.reply_text(
        "👆 Qué respuesta prefieres?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

# ── Inline keyboard callback ───────────────────────────────────────────────────

async def handle_vote(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the user's vote for their preferred answer in eval mode."""
    query = update.callback_query
    await query.answer()  # removes the loading spinner

    # callback_data format: "eval_vote:<user_id>:<mode>"
    _, uid_str, preferred_mode = query.data.split(":", 2)
    user_id = int(uid_str)

    state = user_eval_state.pop(user_id, None)
    if state is None:
        await query.edit_message_text("⚠️ El voto ya se ha guardado o la sesión ha expirado.")
        return

    _log_eval(user_id, state["query"], preferred_mode, state["triple"])

    # Use the preferred answer as the canonical assistant history entry
    chosen_answer = getattr(state["triple"], preferred_mode)
    user_conversations.setdefault(user_id, []).append(
        {"role": "assistant", "content": chosen_answer}
    )

    await query.edit_message_text(
        f"✅ Se ha guardado su preferencia: *{LABELS[preferred_mode]}*\n\nMande su siguiente pregunta.",
        parse_mode="Markdown",
    )

# ── Error Handler ───────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Exception while handling an update:", context.error)

def main():
    load_dotenv(".env")

    hf_client = InferenceClient(provider="novita", api_key=settings.hf_api_key)

    app = ApplicationBuilder().token(settings.telegram_token).build()
    app.bot_data["hf_client"] = hf_client

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("eval", eval_start))
    app.add_handler(CommandHandler("stopeval", eval_stop))
    app.add_handler(CallbackQueryHandler(handle_vote, pattern=r"^eval_vote:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("Bot is running...")
    app.run_polling()
if __name__ == "__main__":
    main()
