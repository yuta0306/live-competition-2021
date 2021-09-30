# For typing
from typing import Union

from transformers import AutoConfig, AutoModel, AutoTokenizer

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

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
    def __init__(self, df, tokenizer, model, kmeans, tsne, max_length: int=64) -> None:
        self.df = df
        self.tokenizer = tokenizer
        self.model = model.to('cpu')
        self.kmeans = kmeans
        self.tsne = tsne
        self.filter = Filter()
        self.max_length = max_length
        self.df_ = df.copy()

    def reply(self, text: str, show_candidate: bool=True, mode='cluster'):
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
        decomposed = self.tsne.transform(last_hidden_state)
        
        # kmeansで最近傍のクラスタを見つける
        label = self.kmeans.predict(decomposed)[0]
        if show_candidate:
            self._show_candidate(label)
        
        if mode == 'distance':
            try:
                responses = self.df['response'].values
                index = self._find_neighbor(decomposed)
                response = responses[index]
            except ValueError:
                self._reset_df()
                responses = self.df['response'].values
                index = self._find_neighbor(decomposed)
                response = responses[index]
        else:
            try:
                candidate_df = self.df[self.df['label'] == label]
                responses = candidate_df['response'].values
                response = responses[np.random.randint(0, responses.shape[0])]
            except ValueError:
                self._reset_df()
                candidate_df = self.df[self.df['label'] == label]
                responses = candidate_df['response'].values
                response = responses[np.random.randint(0, responses.shape[0])]

        self._remove(response)
        # self._filter(response)
        return response

    def _reset_df(self):
        self.df = self.df_.copy()

    def _find_neighbor(self, context_vector):
        # データは2次元圧縮済みのものを期待
        data = self.df.loc[:, ['dim0', 'dim1']].values
        distances = np.linalg.norm(data - context_vector, axis=1)
        index = distances.argmin()
        print('The distance of vector between response and neighbor:', distances[index])
        return index

    def _filter(self, response):
        filtered = self.df['response'].apply(lambda text: self.filter.filter(response, text))
        self.df = self.df[~filtered]

    def _show_candidate(self, label):
        print(f'Prediction >>> {label}')
        candidate_df = self.df[self.df['label'] == label]
        print(candidate_df)

    def _remove(self, text: str):
        self.df = self.df[~(self.df['response'] == text)]


def load_bot(df_path: Union[str, Path], model_name: str, kmeans_path: Union[str, Path],
                tsne_path: Union[str, Path], max_length: int = 64) -> ReplyBot:
    """
    - DataFrameはCSV形式で保存されていること
    - KMeans & TSNEはpickleで保存されていること
    """
    print('LOADING...')
    df = pd.read_csv(df_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    kmeans = pickle.load(open(kmeans_path, 'rb'))
    tsne = pickle.load(open(tsne_path, 'rb'))

    bot = ReplyBot(df, tokenizer, model, kmeans, tsne, max_length)
    print('LOADED!!')
    return bot