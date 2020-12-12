import os
import requests
import telebot
from model import NeuralPoet

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
WELCOME_TEXT = 'welcome text'

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)
poet = NeuralPoet()


def get_poem_from_stirng(message_text):
    neural_poem = poet.predict_on_string(message_text)
    neural_poem = poet.poetize(neural_poem)
    return neural_poem

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, WELCOME_TEXT)

@bot.message_handler(content_types=['text'])
def reply(message):
    reply = get_poem_from_stirng(message.text)
    bot.reply_to(message, reply)


bot.polling()