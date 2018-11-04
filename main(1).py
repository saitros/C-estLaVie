from flask import Flask, request
from pymessenger.bot import Bot
import os



app = Flask(__name__)
ACCESS_TOKEN = ACCESS_TOKEN
VERIFY_TOKEN = VERIFY_TOKEN
bot = Bot(ACCESS_TOKEN)


@app.route('/', methods=['GET', 'POST'])
def receive_message():

    if request.method == 'GET':
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    else:
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                    recipient_id = message['sender']['id']
                    # normal text type input
                    if message['message'].get('text'):
                        print(message)                                                              # get to know
                        response_sent_text = get_message(message['message']['text'])
                        send_message(recipient_id, response_sent_text)
                    # video, image, etc.. non-text type input
                    if message['message'].get('attachments'):
                        response_sent_text = get_message(message['message']['attachments'])
                        send_message(recipient_id, response_sent_text)
    return "Message Processed"

def verify_fb_token(token_sent):
    #take token sent by facebook and verify it matches the verify token
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'

def get_message(message):

    return message

def send_message(recipient_id, response):
    bot.send_text_message(recipient_id, response)
    return "success"




if __name__ == '__main__':

    app.run()



