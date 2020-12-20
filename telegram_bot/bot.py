import requests
import telebot

with open('token.txt', 'r') as f:
    TELEGRAM_TOKEN = f.readline()

WELCOME_TEXT = '''
Я нейросетевой поэт, обученный на плохих стихах из интернета.
Отправь мне любую строку, и я напишу стихотворение на её основе. За качество не ручаюсь.
'''

bot = telebot.TeleBot(TELEGRAM_TOKEN)


def get_poem_from_stirng(message_text):
    response = requests.post('http://0.0.0.0/write_poem', json={'user_string': message_text})
    try:
        answer = response.json()['poem']
    except KeyError:
        try:
            answer = response.json()['error']
        except KeyError:
            answer = 'Unknown error, try again later'
    return answer


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, WELCOME_TEXT)


@bot.message_handler(content_types=['text'])
def reply(message):
    reply = get_poem_from_stirng(message.text)
    bot.reply_to(message, reply)


bot.polling()
