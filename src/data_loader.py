# src/data_loader.py
import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self):
        self.df = None
        self.oh_data = None

    def load_file(self, file_path):
        """엑셀 파일을 읽어옵니다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일을 찾을 수 없습니다.")
        self.df = pd.read_excel(file_path)
        return self.df

    def preprocess(self):
        """데이터를 One-Hot Encoding으로 변환합니다."""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")

        cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        numbers = self.df[cols].values
        
        oh_list = []
        for row in numbers:
            oh_vec = np.zeros(45)
            for num in row:
                oh_vec[int(num)-1] = 1
            oh_list.append(oh_vec)
        
        self.oh_data = np.array(oh_list)
        return self.oh_data

    def get_past_combinations(self):
        """과거 당첨 조합을 세트(Set) 형태로 반환 (중복 체크용)"""
        cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        past_combinations = set()
        for _, row in self.df[cols].iterrows():
            past_combinations.add(tuple(sorted(row)))
        return past_combinations