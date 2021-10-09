# For typing
from typing import Union, Tuple

from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from openTSNE import TSNE

class Filter:
    def __init__(self) -> None:
        pass

    def filter(self, response: str, text: str) -> bool:
        result = False
        if self._member_filter(response):
            result = self._member_filter(text)
        
        return result

    def _member_filter(self, text: str) -> bool:
        members = ['佐藤', '鈴木', '高橋', '渡辺', '小林']
        for member in members:
            if member in text:
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

    def _reset_df(self, name: str):
        df = getattr(self, f'{name}_').copy()
        setattr(self, name, df)

    def _find_neighbor(self, context_vector, name: str) -> Tuple[int, float]:
        # データは2次元圧縮済みのものを期待
        df = getattr(self, name)
        data = df.loc[:, ['dim0', 'dim1']].values
        distances = np.linalg.norm(data - context_vector, axis=1)
        index = distances.argmin()
        print('The distance of vector between response and neighbor:', distances[index])
        return index, distances[index]

    def reply(self, text: str, show_candidate: bool=True):
        """
        text:
            ' [SEP] 'でsysとusrの2発話を結合すること
        """
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
            responses = self.df_context['response'].values
            index, distance = self._find_neighbor(decomposed, 'df_context')
            response = responses[index]
            
        except ValueError:
            self._reset_df('df_context')
            responses = self.df_context['response'].values
            index, distance = self._find_neighbor(decomposed, 'df_context')
            response = responses[index]

        if distance > self.threshold:
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
                index, distance = self._find_neighbor(decomposed, 'df_uttr')
                response = responses[index]
                
            except ValueError:
                self._reset_df('df_uttr')
                responses = self.df_uttr['response'].values
                index, distance = self._find_neighbor(decomposed, 'df_uttr')
                response = responses[index]
        

        if 'サービス' in text and ('?' in text or '？' in text):
            response = 'つまみやお酒を宅配してくれるんです！！'
        self._remove(response)
        self._filter(response)
        return response

    def _filter(self, response):
        filtered = np.array([False for _ in self.df_context['response']])
        if 'オンライン飲み会' in response:
            filtered = self.df_context['response'].apply(self.filter._drink_party_filter)
        self.df_context = self.df_context[~filtered]

        filtered = np.array([False for _ in self.df_uttr['response']])
        if 'オンライン飲み会' in response:
            filtered = self.df_uttr['response'].apply(self.filter._drink_party_filter)
        self.df_uttr = self.df_uttr[~filtered]

    def _remove(self, text: str):
        self.df_context = self.df_context[~(self.df_context['response'] == text)]
        self.df_uttr = self.df_uttr[~(self.df_uttr['response'] == text)]



def load_bot(df_context_path: Union[str, Path], df_uttr_path: Union[str, Path],
            model_name: str, tsne_context_path: Union[str, Path], tsne_uttr_path: Union[str, Path],
            max_length: int = 64) -> ReplyBot:
    """
    - DataFrameはCSV形式で保存されていること
    - KMeans & TSNEはpickleで保存されていること
    """
    print('LOADING...')
    df_context = pd.read_csv(df_context_path)
    df_uttr = pd.read_csv(df_uttr_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tsne_context = pickle.load(open(tsne_context_path, 'rb'))
    tsne_uttr = pickle.load(open(tsne_uttr_path, 'rb'))

    bot = ReplyBot(df_context=df_context, df_uttr=df_uttr,tokenizer=tokenizer,
                    model=model, tsne_context=tsne_context, tsne_uttr=tsne_uttr,
                    max_length=max_length)
    print('LOADED!!')
    return bot