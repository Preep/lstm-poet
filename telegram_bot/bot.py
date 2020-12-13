import os
import requests
import telebot

with open('token.txt', 'r') as f:
    TELEGRAM_TOKEN = f.readline()
print(TELEGRAM_TOKEN)
WELCOME_TEXT = 'welcome text'

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)


def get_poem_from_stirng(message_text):
    response = requests.post(
        'http://localhost/write_poem',
        json={'user_string': message_text}
    )
    poem = response.json['poem']
    return poem


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, WELCOME_TEXT)


@bot.message_handler(content_types=['text'])
def reply(message):
    reply = get_poem_from_stirng(message.text)
    bot.reply_to(message, reply)


bot.polling()
