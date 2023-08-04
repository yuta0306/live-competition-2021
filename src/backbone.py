# For typing
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, BertModel


class Filter:
    def __init__(self):
        pass

    def filter(self, response: str, text: str):
        result = False
        if self._member_filter(response):
            result = self._member_filter(text)

        return result

    def _member_filter(self, text: str):
        members = ["佐藤", "鈴木", "高橋", "渡辺", "小林"]  # 佐藤、鈴木、高橋、渡辺、小林
        for member in members:
            if member in text:
                return True
        return False

    def _drink_party_filter(self, text: str):
        filtered_word = "オンライン飲み会"
        if filtered_word in text:
            return True
        return False


# ReplyBot
class ReplyBot:
    def __init__(
        self,
        df_context: pd.DataFrame,
        df_uttr: pd.DataFrame,
        tokenizer,
        model: BertModel,
        context_emb: np.ndarray,
        uttr_emb: np.ndarray,
        max_length: int = 64,
        threshold: float = 1.0,
    ):
        self.df_context = df_context
        self.df_uttr = df_uttr
        self.tokenizer = tokenizer
        self.model = model.to("cpu")
        self.context_emb = context_emb
        self.uttr_emb = uttr_emb
        self.filter = Filter()
        self.max_length = max_length
        self.threshold = threshold

        self.df_context_ = df_context.copy()
        self.df_uttr_ = df_uttr.copy()

        self.df_by_id: Dict[int, Dict[str, pd.DataFrame]] = {}

        # ルールベースパラメータ
        self.rulebase_params: Dict[int, Dict[str, Any]] = {}

    def register_chat_id(self, id: int):
        init_params = {
            "explained_delivery": False,
            "explained_member": False,
            "explained_other_member": False,
            "new_topic": [
                "あ！そういえば今回の飲み会、新しいオンラインサービスを使ってみようと思ってるんですよ！",
                "今回は先輩も交えて楽しく飲みたいね～ってみんなで話してたんですけど…",
                "親睦をさらに深めたいんです！お願いします！",
            ],
        }
        self.rulebase_params[id] = init_params
        self.df_by_id[id] = {
            "df_context": self.df_context,
            "df_uttr": self.df_uttr,
        }

        # 擬似的に先にフィルタリング
        self._filter("オンライン飲み会", id=id)

    def _reset_df(self, name: str, id: int):
        df = getattr(self, f"{name}_").copy()
        self.df_by_id[id][name] = df

    def _find_neighbor(
        self,
        context_vector: np.ndarray,
        type_: str,
    ) -> Tuple[int, float]:
        # データは2次元圧縮済みのものを期待
        if type_ == "context":
            data = self.context_emb
        else:
            data = self.uttr_emb
        distances = np.linalg.norm(data - context_vector, axis=1)
        index = distances.argmin()
        print("The distance of vector between response and neighbor:", distances[index])
        return index, distances[index]

    def reply(self, text: str, id: int, show_candidate: bool = True):
        """
        text:
            ' [SEP] 'でsysとusrの2発話を結合すること
        """
        (
            explained_delivery,
            explained_member,
            explained_other_member,
            *_,
        ) = self.rulebase_params[id].values()

        print("Input:", text)
        encoded = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            bert_output = self.model(**encoded)
        last_hidden_state = bert_output.last_hidden_state
        last_hidden_state = last_hidden_state[:, 0, :].numpy()

        distance = 0.0
        try:
            index, distance = self._find_neighbor(last_hidden_state, "context")
            response = self.df_context_["response"][index]
            print("↑", response)

        except ValueError:
            self._reset_df("df_context", id=id)
            index, distance = self._find_neighbor(last_hidden_state, "context")
            response = self.df_context_["response"][index]
            print("↑", response)

        if distance > self.threshold:
            distance_context = distance
            response_context = response
            text = text.split(" [SEP] ")[-1]
            print("Input:", text)
            encoded = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                bert_output = self.model(**encoded)
            last_hidden_state = bert_output.last_hidden_state
            last_hidden_state = last_hidden_state[:, 0, :].numpy()
            try:
                index, distance = self._find_neighbor(last_hidden_state, "uttr")
                response = self.df_context_["response"][index]
                print("↑", response)

            except ValueError:
                self._reset_df("df_uttr", id=id)
                index, distance = self._find_neighbor(last_hidden_state, "uttr")
                response = self.df_context_["response"][index]
                # print('↑', response)

            if distance_context < distance:
                response = response_context
                distance = distance_context

        # ルールベース処理
        text = text.split(" [SEP] ")[-1]
        # print('Start Filtering:', text)
        if "サービス" in text and ("?" in text or "？" in text):
            # print('< Delivery Filter >')
            response = "つまみやお酒を宅配してくれるんです！！" if not explained_delivery else response
            self.rulebase_params[id]["explained_delivery"] = True
        elif (
            ("他" in text or "ほか" in text or "以外" in text)
            and ("?" in text or "？" in text)
            and explained_member
        ):
            # print('< Other Member Filter >')
            response = "いえ、このメンバーしか誘ってません" if not explained_other_member else response
            self.rulebase_params[id]["explained_other_member"] = True
        elif "誰" in text or "だれ" in text or "メンバ" in text:
            # print('< Member Filter >')
            response = "佐藤、鈴木、高橋、渡辺、小林がきます！" if not explained_member else response
            self.rulebase_params[id]["explained_member"] = True
            self._member_filter(id=id)
        elif "6" in text:
            response = "湯川先輩を入れれば7人ですね"
        else:
            # print('< No Filter >')
            if (
                distance > self.threshold
                and len(self.rulebase_params[id]["new_topic"]) > 0
            ):
                # print(distance)
                response = self.rulebase_params[id]["new_topic"].pop(0)

        self._remove(response, id=id)
        self._filter(response, id=id)
        self._typing(response=response)
        print(f"Final Response: {response}")
        return response

    def _filter(self, response, id: int):
        filtered = np.array(
            [False for _ in self.df_by_id[id]["df_context"]["response"]]
        )
        if "オンライン飲み会" in response:
            filtered = self.df_by_id[id]["df_context"]["response"].apply(
                self.filter._drink_party_filter
            )
        self.df_by_id[id]["df_context"] = self.df_by_id[id]["df_context"][~filtered]

        filtered = np.array([False for _ in self.df_by_id[id]["df_uttr"]["response"]])
        if "オンライン飲み会" in response:
            filtered = self.df_by_id[id]["df_uttr"]["response"].apply(
                self.filter._drink_party_filter
            )
        self.df_by_id[id]["df_uttr"] = self.df_by_id[id]["df_uttr"][~filtered]

    def _member_filter(self, id: int):
        filtered = np.array(
            [False for _ in self.df_by_id[id]["df_context"]["response"]]
        )
        filtered = self.df_by_id[id]["df_context"]["response"].apply(
            self.filter._member_filter
        )
        self.df_by_id[id]["df_context"] = self.df_by_id[id]["df_context"][~filtered]

        filtered = np.array([False for _ in self.df_by_id[id]["df_uttr"]["response"]])
        filtered = self.df_by_id[id]["df_uttr"]["response"].apply(
            self.filter._member_filter
        )
        self.df_by_id[id]["df_uttr"] = self.df_by_id[id]["df_uttr"][~filtered]

    def _typing(self, response: str):
        length = len(response)
        avg_time = length / 4
        typing_time = avg_time + (random.random() * avg_time * 1 / 3)
        time.sleep(typing_time)

    def _remove(self, text: str, id: int):
        self.df_by_id[id]["df_context"] = self.df_by_id[id]["df_context"][
            ~(self.df_by_id[id]["df_context"]["response"] == text)
        ]
        self.df_by_id[id]["df_uttr"] = self.df_by_id[id]["df_uttr"][
            ~(self.df_by_id[id]["df_uttr"]["response"] == text)
        ]


def load_bot(
    df_context_path: Union[str, Path],
    df_uttr_path: Union[str, Path],
    model_name: str,
    context_emb_path: Union[str, Path],
    uttr_emb_path: Union[str, Path],
    max_length: int = 64,
    threshold: float = 1.0,
) -> ReplyBot:
    """
    - DataFrameはCSV形式で保存されていること
    - KMeans & TSNEはpickleで保存されていること
    """
    print("LOADING...")
    df_context = pd.read_csv(df_context_path).reset_index().set_index("index")
    df_uttr = pd.read_csv(df_uttr_path).reset_index().set_index("index")
    print("COMPLETED LOADING DATAFRAMES.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("COMPLETED LOADING TOKENIZER.")
    model = AutoModel.from_pretrained(model_name)
    print("COMPLETED LOADING BERT MODEL.")
    context_emb = np.load(context_emb_path)
    uttr_emb = np.load(uttr_emb_path)
    print("COMPLETED LOADING TSNE EMBEDDING.")

    bot = ReplyBot(
        df_context=df_context,
        df_uttr=df_uttr,
        tokenizer=tokenizer,
        model=model,
        context_emb=context_emb,
        uttr_emb=uttr_emb,
        max_length=max_length,
        threshold=threshold,
    )
    print("COMPLETED CREATING REPLY BOT.")
    print("LOADED!!")
    return bot
