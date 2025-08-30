from telegram import Update, Bot
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

TELEGRAM_BOT_TOKEN = '8177282153:AAHf7HJlNwUG23JMJa9dvnEstihel68VjPU'

# This handler will echo your chat ID when you send any message to the bot

def echo_chat_id(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id=chat_id, text=f"Your chat ID is: {chat_id}")
    print(f"Received message from chat ID: {chat_id}")

def main():
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), echo_chat_id))
    print("Bot is running. Send a message to your bot in Telegram.")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
