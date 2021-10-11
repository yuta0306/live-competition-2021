# For typing
from re import L
from typing import Union, Tuple, Dict
import time
import random

from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from openTSNE import TSNE

class Filter:
    def __init__(self):
        pass

    def filter(self, response: str, text: str):
        result = False
        if self._member_filter(response):
            result = self._member_filter(text)

        return result

    def _member_filter(self, text: str):
        members = ['佐藤', '鈴木', '高橋', '渡辺', '小林']  # 佐藤、鈴木、高橋、渡辺、小林
        for member in members:
            if member in text:
                return True
        return False

    def _drink_party_filter(self, text: str):
        filtered_word = 'オンライン飲み会'
        if filtered_word in text:
            return True
        return False

# ReplyBot
class ReplyBot:
    def __init__(self, df_context: pd.DataFrame, df_uttr : pd.DataFrame, tokenizer, model: BertModel,
                 tsne_context: TSNE, tsne_uttr: TSNE, max_length: int = 64, threshold: float = 1.0):
        self.df_context = df_context
        self.df_uttr = df_uttr
        self.tokenizer = tokenizer
        self.model = model.to('cpu')
        self.tsne_context = tsne_context
        self.tsne_uttr = tsne_uttr
        self.filter = Filter()
        self.max_length = max_length
        self.threshold = threshold

        self.df_context_ = df_context.copy()
        self.df_uttr_ = df_uttr.copy()

        self.df_by_id: Dict[int, Dict[str, pd.DataFrame]] = {}

        # ルールベースパラメータ
        self.rulebase_params: Dict[int, Dict[str, bool]] = {}
        
        # 擬似的に先にフィルタリング
        self._filter('オンライン飲み会')

    def register_chat_id(self, id: int):
        init_params = {
            'explained_delivery': False,
            'explained_member': False,
            'explained_other_member': False,
            'new_topic': [
                'あ！そういえば今回の飲み会、新しいオンラインサービスを使ってみようと思ってるんですよ！',
                '今回は先輩も交えて楽しく飲みたいね～ってみんなで話してたんですけど…',
                '親睦をさらに深めたいんです！お願いします！',
            ],
        }
        self.rulebase_params[id] = init_params
        self.df_by_id[id] = {
            'df_context': self.df_by_id,
            'df_uttr': self.df_uttr,
        }

    def _reset_df(self, name: str, id: int):
        df = getattr(self, f'{name}_').copy()
        self.df_by_id[id][name] = df

    def _find_neighbor(self, context_vector, name: str, id: int) -> Tuple[int, float]:
        # データは2次元圧縮済みのものを期待
        df = self.df_by_id[id][name]
        data = df.loc[:, ['dim0', 'dim1']].values
        distances = np.linalg.norm(data - context_vector, axis=1)
        index = distances.argmin()
        print('The distance of vector between response and neighbor:', distances[index])
        return index, distances[index]

    def reply(self, text: str, id: int, show_candidate: bool=True):
        """
        text:
            ' [SEP] 'でsysとusrの2発話を結合すること
        """
        explained_delivery, explained_member, explained_other_member, *_ = self.rulebase_params[id].values()

        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length, 
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids, token_type_ids, attention_mask = encoded.values()
        bert_output = self.model(input_ids, token_type_ids, attention_mask)
        last_hidden_state = bert_output.last_hidden_state
        last_hidden_state = last_hidden_state.mean(dim=1).detach().numpy()
        decomposed = self.tsne_context.transform(last_hidden_state)
        
        distance = 0
        try:
            responses = self.df_by_id[id]['df_context']['response'].values
            index, distance = self._find_neighbor(decomposed, 'df_context', id=id)
            response = responses[index]
            print('↑', response)
            
        except ValueError:
            self._reset_df('df_context', id=id)
            responses = self.df_by_id[id]['df_context']['response'].values
            index, distance = self._find_neighbor(decomposed, 'df_context', id=id)
            response = responses[index]
            print('↑', response)

        if distance > self.threshold:
            distance_context = distance
            response_context = response
            text = text.split(' [SEP] ')[-1]
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length, 
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids, token_type_ids, attention_mask = encoded.values()
            bert_output = self.model(input_ids, token_type_ids, attention_mask)
            last_hidden_state = bert_output.last_hidden_state
            last_hidden_state = last_hidden_state.mean(dim=1).detach().numpy()
            decomposed = self.tsne_uttr.transform(last_hidden_state)
            try:
                responses = self.df_uttr['response'].values
                index, distance = self._find_neighbor(decomposed, 'df_uttr', id=id)
                response = responses[index]
                print('↑', response)
                
            except ValueError:
                self._reset_df('df_uttr', id=id)
                responses = self.df_uttr['response'].values
                index, distance = self._find_neighbor(decomposed, 'df_uttr', id=id)
                response = responses[index]
                print('↑', response)
            
            if distance_context < distance:
                response = response_context
                distance = distance_context
                
        # ルールベース処理
        text = text.split(' [SEP] ')[-1]
        print('Start Filtering:', text)
        if 'サービス' in text and ('?' in text or '？' in text):
            print('< Delivery Filter >')
            response = 'つまみやお酒を宅配してくれるんです！！' if not explained_delivery else response
            self.rulebase_params[id]['explained_delivery'] = True
        elif ('他' in text or 'ほか' in text or '以外' in text) and ('?' in text or '？' in text) and explained_member:
            print('< Other Member Filter >')
            response = 'いえ、このメンバーしか誘ってません' if not explained_other_member else response
            self.rulebase_params[id]['explained_other_member'] = True
        elif ('誰' in text or 'だれ' in text or 'メンバ' in text):
            print('< Member Filter >')
            print(explained_member)
            response = '佐藤、鈴木、高橋、渡辺、小林がきます！' if not explained_member else response
            self.rulebase_params[id]['explained_member'] = True
            self._member_filter(id=id)
        elif '6' in text:
            response = '湯川先輩を入れれば7人ですね'
        else:
            print('< No Filter >')
            if distance > self.threshold and len(self.rulebase_params[id]['new_topic']) > 0:
                print(distance)
                response = self.rulebase_params[id]['new_topic'].pop(0)

        self._remove(response, id=id)
        self._filter(response, id=id)
        self._typing(response=response)
        print(f'Final Response: {response}')
        return response

    def _filter(self, response, id: int):
        filtered = np.array([False for _ in self.df_by_id[id]['df_context']['response']])
        if 'オンライン飲み会' in response:
            filtered = self.df_by_id[id]['df_context']['response'].apply(self.filter._drink_party_filter)
        self.df_by_id[id]['df_context'] = self.df_by_id[id]['df_context'][~filtered]

        filtered = np.array([False for _ in self.df_by_id[id]['df_uttr']['response']])
        if 'オンライン飲み会' in response:
            filtered = self.df_by_id[id]['df_uttr']['response'].apply(self.filter._drink_party_filter)
        self.df_by_id[id]['df_uttr'] = self.df_by_id[id]['df_uttr'][~filtered]

    def _member_filter(self, id: int):
        filtered = np.array([False for _ in self.df_by_id[id]['df_context']['response']])
        filtered = self.df_by_id[id]['df_context']['response'].apply(self.filter._member_filter)
        self.df_by_id[id]['df_context'] = self.df_by_id[id]['df_context'][~filtered]

        filtered = np.array([False for _ in self.df_by_id[id]['df_uttr']['response']])
        filtered = self.df_by_id[id]['df_uttr']['response'].apply(self.filter._member_filter)
        self.df_by_id[id]['df_uttr'] = self.df_by_id[id]['df_uttr'][~filtered]

    def _typing(self, response: str):
        length = len(response)
        avg_time = length / 4
        typing_time = avg_time + (random.random() * avg_time * 1/3)
        time.sleep(typing_time)

    def _remove(self, text: str, id: int):
        self.df_by_id[id]['df_context'] = self.df_by_id[id]['df_context'][~(self.df_by_id[id]['df_context']['response'] == text)]
        self.df_by_id[id]['df_uttr'] = self.df_by_id[id]['df_uttr'][~(self.df_by_id[id]['df_uttr']['response'] == text)]



def load_bot(df_context_path: Union[str, Path], df_uttr_path: Union[str, Path],
            model_name: str, tsne_context_path: Union[str, Path], tsne_uttr_path: Union[str, Path],
            max_length: int = 64, threshold: float = 1.0) -> ReplyBot:
    """
    - DataFrameはCSV形式で保存されていること
    - KMeans & TSNEはpickleで保存されていること
    """
    print('LOADING...')
    df_context = pd.read_csv(df_context_path)
    df_uttr = pd.read_csv(df_uttr_path)
    print('COMPLETED LOADING DATAFRAMES.')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('COMPLETED LOADING TOKENIZER.')
    model = AutoModel.from_pretrained(model_name)
    print('COMPLETED LOADING BERT MODEL.')
    tsne_context = pickle.load(open(tsne_context_path, 'rb'))
    tsne_uttr = pickle.load(open(tsne_uttr_path, 'rb'))
    print('COMPLETED LOADING TSNE EMBEDDING.')

    bot = ReplyBot(df_context=df_context, df_uttr=df_uttr,tokenizer=tokenizer,
                    model=model, tsne_context=tsne_context, tsne_uttr=tsne_uttr,
                    max_length=max_length, threshold=threshold)
    print('COMPLETED CREATING REPLY BOT.')
    print('LOADED!!')
    return bot