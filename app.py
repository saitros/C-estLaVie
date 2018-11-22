from flask import Flask,request
from pymessenger.bot import Bot
import api_call
import os



app = Flask(__name__)
ACCESS_TOKEN = "EAAEy9F39mZCUBAKC35TVZCOgj2GJ7n0wVIVaZBT0KtGOJa85p7g5l22mDlTnF2Ey8onG6ZAblygheZCnOGeygUlX8kk5QomoUKKZCflIxU2J3D9WqWhzE5aCMF7fFIiDDKc7CsF9bDo9Y7E5iOz2T98OFT6ogKzBx8muPD3bJC9A1FYU9eCom1"
VERIFY_TOKEN = "CestLaVie"
bot = Bot(ACCESS_TOKEN)

@app.route('/')
def hello_world():
    return 'Hello World!'

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
                        # intent model & entity model ==> api call ==> msg make                     # get str
                        # response = get_message(message['message']['text'])
                        response_sent_text = get_message(message['message']['text']) # replace response
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
    app.run(host="0.0.0.0", port = 5000)
