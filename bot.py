from typing import NoReturn
from telegram.ext import (
    Updater, CommandHandler, MessageHandler, Filters, CallbackContext,
)
from telegram.update import Update
import time
import random

from src.utils import read_config_file

CONFIG: dict = read_config_file('.telegram')

class YuBot:
    def __init__(self) -> None:
        self.config = CONFIG
        self.user_context: dict = {}
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
        if update.message.from_user.id not in self.user_context:
            self.user_context[update.message.from_user.id] = {"context": [], "count": 0}

        # ユーザ発話の回数を更新
        self.user_context[update.message.from_user.id]["count"] += 1

        # ユーザ発話をcontextに追加
        self.user_context[update.message.from_user.id]["context"].append(update.message.text)

        # replyメソッドによりcontextから発話を生成
        send_message = self._reply(self.user_context[update.message.from_user.id]["context"])

        # 送信する発話をcontextに追加
        self.user_context[update.message.from_user.id]["context"].append(send_message)

        # 発話を送信
        update.message.reply_text(send_message)

        if self.user_context[update.message.from_user.id]["count"] >= self.config['DIALOGUE_LENGTH']:
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