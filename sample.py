#!/usr/bin/env python
# -*- coding: utf-8 -*-

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import time
import random

# アクセストークン
TOKEN = ""

# 1対話の長さ(ユーザの発話回数)．ここは固定とする
DIALOGUE_LENGTH = 15

"""現在の対話のログからランダムに発話を選択し，送信するボット"""

# 対話履歴を受け取り，応答を返す．
# ここを各自書き換えれば自分のシステムができる
# このサンプルでは対話履歴の中からランダムで選択し，それを応答とする．
def reply(context):
    return random.choice(context)


class SampleBot:
    def __init__(self):
        self.user_context = {}

    def start(self, bot, update):
        # 対話ログと発話回数を初期化
        self.user_context[update.message.from_user.id] = {"context": [], "count": 0}

        # システムからの最初の発話
        # 以下の発話に限定しません．任意の発話を返してください
        update.message.reply_text('こんにちは。対話を始めましょう。')


    def message(self, bot, update):
        if update.message.from_user.id not in self.user_context:
            self.user_context[update.message.from_user.id] = {"context": [], "count": 0}

        # ユーザ発話の回数を更新
        self.user_context[update.message.from_user.id]["count"] += 1

        # ユーザ発話をcontextに追加
        self.user_context[update.message.from_user.id]["context"].append(update.message.text)

        # replyメソッドによりcontextから発話を生成
        send_message = reply(self.user_context[update.message.from_user.id]["context"])

        # 送信する発話をcontextに追加
        self.user_context[update.message.from_user.id]["context"].append(send_message)

        # 発話を送信
        update.message.reply_text(send_message)

        if self.user_context[update.message.from_user.id]["count"] >= DIALOGUE_LENGTH:
            # 対話IDは unixtime:user_id:bot_username
            unique_id = str(int(time.mktime(update.message["date"].timetuple()))) + u":" + str(update.message.from_user.id) + u":" + bot.username

            update.message.reply_text(u"_FINISHED_:" + unique_id)
            update.message.reply_text(u"対話終了です．エクスポートした「messages.html」ファイルを，フォームからアップロードしてください．")


    def run(self):
        updater = Updater(TOKEN)

        dp = updater.dispatcher

        dp.add_handler(CommandHandler("start", self.start))

        dp.add_handler(MessageHandler(Filters.text, self.message))

        updater.start_polling()

        updater.idle()


if __name__ == '__main__':
    mybot = SampleBot()
    mybot.run()
