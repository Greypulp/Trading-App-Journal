from telegram import Bot, Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
import threading
import time

TELEGRAM_BOT_TOKEN = '8177282153:AAHf7HJlNwUG23JMJa9dvnEstihel68VjPU'
TELEGRAM_CHAT_ID = '8426349009'  # integer or string

approval_event = threading.Event()
approval_result = {'approved': False}

bot = Bot(token=TELEGRAM_BOT_TOKEN)

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text.lower().strip()
    if text == 'approve':
        approval_result['approved'] = True
        approval_event.set()
        context.bot.send_message(chat_id=update.effective_chat.id, text='Approved!')
    elif text == 'reject':
        approval_result['approved'] = False
        approval_event.set()
        context.bot.send_message(chat_id=update.effective_chat.id, text='Rejected!')

def main():
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), handle_message))
    # Send approval request
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="Test approval: Reply with 'approve' or 'reject'.")
    print('Waiting for approval in Telegram (reply approve/reject)...')
    approval_event.clear()
    # Start polling (blocking)
    t = threading.Thread(target=updater.start_polling, daemon=True)
    t.start()
    approved = approval_event.wait(timeout=120)
    if approved:
        if approval_result['approved']:
            print('Test: Approved by user.')
        else:
            print('Test: Rejected by user.')
    else:
        print('Test: No response received in 2 minutes.')
    updater.stop()

if __name__ == "__main__":
    main()
