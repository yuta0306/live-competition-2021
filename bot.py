import re
from typing import NoReturn
from telegram import message
from telegram.ext import (
    Updater, CommandHandler, MessageHandler, Filters, CallbackContext,
)
from telegram.update import Update
import time
import random

# Sentiment Analysis
from transformers import pipeline
from transformers import T5Tokenizer, AutoModelForCausalLM

from src.utils import read_config_file

CONFIG: dict = read_config_file('.telegram')

class YuBot:
    def __init__(self) -> None:
        self.config = CONFIG
        self.user_context: dict = {}
        # self.model = pipeline("sentiment-analysis",model="daigo/bert-base-japanese-sentiment",tokenizer="daigo/bert-base-japanese-sentiment")
        self.tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        self.tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        self.tokenizer.add_special_tokens({'additional_special_tokens' : ['<sys>', '<usr>']})

        self.model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self._validate_config()

    def _validate_config(self) -> NoReturn:
        if not getattr(self, 'config'):
            raise ValueError
        if not ('TOKEN' in self.config.keys() and 
                'DIALOGUE_LENGTH' in self.config.keys()):
            raise ValueError
        if not (isinstance(self.config['TOKEN'], str) and
                isinstance(self.config['DIALOGUE_LENGTH'], int)):
            raise ValueError

    def start(self, update: Update, context: CallbackContext) -> NoReturn:
        # 対話ログと発話回数を初期化
        self.user_context[update.message.from_user.id] = {"context": [], "count": 0}

        # システムからの最初の発話
        # 以下の発話に限定しません．任意の発話を返してください
        update.message.reply_text('湯川先輩，お疲れ様です!!😄')

    def _reply(self, context):
        return random.choice(context)

    def message(self, update: Update, context: CallbackContext):
        if update.message.from_user.id not in self.user_context or 'history' not in self.user_context[update.message.from_user.id].keys():
            self.user_context[update.message.from_user.id] = {"context": [], "count": 0, "history": '<sys>湯川先輩，お疲れ様です!!'}

        # ユーザ発話の回数を更新 && Turnを保持
        self.user_context[update.message.from_user.id]["count"] += 1
        turns = self.user_context[update.message.from_user.id]["count"]

        # ユーザ発話をcontextに追加
        self.user_context[update.message.from_user.id]["context"].append(update.message.text)
        user_message = update.message.text
        self.user_context[update.message.from_user.id]["history"] += '<usr>' + user_message

        # replyメソッドによりcontextから発話を生成
        send_message = self._reply(self.user_context[update.message.from_user.id]["context"])

        # 送信する発話をcontextに追加
        self.user_context[update.message.from_user.id]["context"].append(send_message)

        # 発話を送信
        if turns == 1:
            update.message.reply_text('突然連絡してすみません．')
            self.user_context[update.message.from_user.id]["history"] += '<sys>突然連絡してすみません．'
        elif turns == 2:
            update.message.reply_text('そういえば，週末暇だってボヤいてましたよね👀')
            self.user_context[update.message.from_user.id]["history"] += '<sys>そういえば，週末暇だってボヤいてましたよね．'

        elif turns == 3:
            msg = ''
            # if self.model(user_message)[0]['score'] <s .8:
            #     msg = '暇って言ってましたよ〜．'
            update.message.reply_text(msg + '今度の週末にオンライン飲み会するんですが，一緒にどうですか？')
            self.user_context[update.message.from_user.id]["history"] += '<sys>今度の週末にオンライン飲み会するんですが，一緒にどうですか？'
        elif turns > 3:
            if '誰' in user_message:
                update.message.reply_text('佐藤，鈴木，高橋，渡辺，小林が今のところ来れるみたいです．')
                self.user_context[update.message.from_user.id]["history"] += '<sys>突然連絡してすみません．'
            elif '同期' in user_message:
                update.message.reply_text('同期との飲みですが，先輩には仲良くしてもらってますし，同期のみんなからの信頼も厚いですから．．．')
                self.user_context[update.message.from_user.id]["history"] += '<sys>同期との飲みですが，先輩には仲良くしてもらってますし，同期のみんなからの信頼も厚いですから．．．'
            else:
                input_ = self.tokenizer.encode(self.user_context[update.message.from_user.id]["history"], return_tensors="pt")
                output = self.model.generate(input_, do_sample=True, max_length=64, num_return_sequences=1)
                output = self.tokenizer.batch_decode(output)
                print(output)
                msg = output[0].split('</s>')[-1]
                update.message.reply_text(msg)
                self.user_context[update.message.from_user.id]["history"] += msg

        else:
            update.message.reply_text(send_message)

        if self.user_context[update.message.from_user.id]["count"] >= self.config['DIALOGUE_LENGTH'] // 2:
            # 対話IDは unixtime:user_id:bot_username
            unique_id = str(int(time.mktime(update.message["date"].timetuple()))) + u":" + str(update.message.from_user.id) + u":" + context.bot.username

            update.message.reply_text(u"_FINISHED_:" + unique_id)
            update.message.reply_text(u"対話終了です．エクスポートした「messages.html」ファイルを，フォームからアップロードしてください．")

    def run(self) -> NoReturn:
        updater = Updater(self.config['TOKEN'], use_context=True)

        dp = updater.dispatcher

        dp.add_handler(CommandHandler("start", self.start))

        dp.add_handler(MessageHandler(Filters.text, self.message))

        updater.start_polling()

        updater.idle()

if __name__ == '__main__':
    bot = YuBot()
    bot.run()