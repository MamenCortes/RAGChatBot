import os
import csv
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, UTC
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from huggingface_hub import InferenceClient
import random

from .config import settings
from .rag import rag_answer, rag_answer_3_modes

datetime.now(UTC).isoformat()

user_conversations: dict[int, list[dict]] = {}

# Eval state per user: stores the last TripleAnswer while waiting for a vote
user_eval_state: dict[int, dict] = {}
# Set of user_ids currently in eval mode
eval_mode_users: set[int] = set()

EVAL_LOG_PATH = Path("eval/eval_results.csv")
#MODES = ["semantic", "hybrid", "language_aware_hybrid", "no_retrieval"]
MODES = ["system_prompt_only", "no_retrieval", "rag_retrieval"] # must match TripleAnswer fields
LABELS = {
    "system_prompt_only": "Solo prompt del sistema",
    "no_retrieval": "Sin recuperación (solo LLM)",
    "rag_retrieval": "RAG híbrido (semántica + keyword)",
}
#LABELS = {
#    "semantic":     "Semantic",
#    "hybrid":       "Hybrid (Semantic + Keyword)",
#    "language_aware_hybrid":    "Language-aware Hybrid",
#    "no_retrieval": "No retrieval (LLM only)",}

# ── CSV helpers ────────────────────────────────────────────────────────────────

def _ensure_csv() -> None:
    """Ensure the CSV log file exists and has the correct header."""
    if not EVAL_LOG_PATH.exists():
        print(f"Creating new evaluation log at {EVAL_LOG_PATH}")
        with open(EVAL_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            #Revised, updated header to include preferred_mode and remove individual mode columns
            writer.writerow([
                "timestamp",
                "user_id",
                "query",
                "preferred_mode",
                "answer_system_prompt_only",
                "answer_no_retrieval",
                "answer_rag_retrieval",
            ])
        return

    print(f"Evaluation log ready at {EVAL_LOG_PATH}")


def _log_eval(user_id: int, query: str, preferred_mode: str, multiple) -> None:
    """Append an evaluation result to the CSV log."""
    _ensure_csv()
    with open(EVAL_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(UTC).isoformat(),
            user_id,
            query,
            preferred_mode,
            multiple.system_prompt_only,
            multiple.no_retrieval,
            multiple.rag_retrieval,
        ])
    print(f"Logged evaluation: user_id={user_id}, query='{query}', preferred_mode='{preferred_mode}'")

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
    print(f"User {user_id} has entered evaluation mode.")
    await update.message.reply_text(
    "📊 *Modo de evaluación ACTIVADO*\n\n"
    "Para cada consulta generaré varias respuestas utilizando diferentes estrategias de recuperación:\n"
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
    print(f"User {user_id} has exited evaluation mode.")
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

    print(f"Received message from user {user_id}: {text}")

    # ── Normal mode ──────────────────────────────────────────────────────────
    if user_id not in eval_mode_users:
        await update.message.reply_text("Pensando... 🤔")
        try:
            answer = rag_answer(
                hf_client=hf_client,
                user_message=text,
                chat_history=history,
                top_k=settings.top_k,
                model=settings.llm_model_name,
            )
            #Add the new question and answer to the history.
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": answer})
            await update.message.reply_text(answer)
        except Exception as e:
            print(f"Error: {e}")
            await update.message.reply_text("¡Algo ha salido mal! 😥")
        return

    # ── Eval mode ────────────────────────────────────────────────────────────
    await update.message.reply_text("Pensando... 🤔")
    # It is not necessary to store the user messages in the history for eval mode, since the user will be voting on which answer they 
    # prefer among the multiple generated answers for the same query, and we want to avoid contaminating the history with multiple user 
    # turns that are essentially the same question. 
    # Instead, we can just pass the current query as a parameter to rag_answer_3_modes without appending it to the history. 
    # The history can still contain previous turns from before eval mode was activated, 
    # which provides context to the LLM without being cluttered by repeated user queries during eval mode.
    #history.append({"role": "user", "content": text})

    try:
        multiple = rag_answer_3_modes(
            hf_client=hf_client,
            user_message=text,
            top_k=settings.top_k,
            model=settings.llm_model_name,
        )
    except Exception as e:
        print(f"Error: {e}")
        await update.message.reply_text("¡Algo ha salido mal! 😥")
        return

    # Store state so the callback can access query + triple
    user_eval_state[user_id] = {"query": text, "multiple": multiple}

    
    #answers = {
    #    "semantic":     multiple.semantic,
    #    "hybrid":       multiple.hybrid,
    #    "language_aware_hybrid": multiple.language_aware_hybrid,
    #    "no_retrieval": multiple.no_retrieval,}

    # Send the three answers
    answers = {
        "system_prompt_only": multiple.system_prompt_only,
        "no_retrieval": multiple.no_retrieval,
        "rag_retrieval": multiple.rag_retrieval}
    
    #Shuffle the order of the answers to avoid position bias in the evaluation. The callback data still contains the mode, so we can identify which answer was chosen regardless of the order they are displayed.
    items = list(answers.items())
    random.shuffle(items)

    for i, (mode, answer) in enumerate(items, start=1):
        print(f"Answer {i} for mode {mode}:\n{answer[:100]}\n")
        await update.message.reply_text(
            #f"{i}. *{LABELS[mode]}*\n\n{answer}",
            f"Respuesta {i}. \n\n{answer}",
        )
        #parse_mode="Markdown",

    # Inline keyboard with one button per answer
    #keyboard = [[
    #    InlineKeyboardButton(f"{i}. {LABELS[mode]}", callback_data=f"eval_vote:{user_id}:{mode}")
    #    for i, mode in enumerate(MODES, start=1)]]

    # Build buttons in the same shuffled order,
    # but store the true mode in callback_data
    keyboard = [[
        InlineKeyboardButton(
            f"Respuesta {i}",
            callback_data=f"eval_vote:{user_id}:{mode}"
        )
        for i, (mode, _) in enumerate(items, start=1)
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

    print(f"User {user_id} voted for mode {preferred_mode} on query: {state['query']}")
    _log_eval(user_id, state["query"], preferred_mode, state["multiple"])

    #Not need to save the preferred answer in the conversation history. 
    # Use the preferred answer as the canonical assistant history entry
    #chosen_answer = getattr(state["triple"], preferred_mode)
    #user_conversations.setdefault(user_id, []).append(
    #    {"role": "assistant", "content": chosen_answer})

    await query.edit_message_text(
        f"✅ Se ha guardado su preferencia.\n\n Mande su siguiente pregunta.",
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
