# src/ai_engine.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LottoAI:
    def __init__(self):
        self.model = None
        self.window_size = 5

    def create_dataset(self, oh_data):
        """시계열 데이터셋 생성"""
        x_data, y_data = [], []
        for i in range(len(oh_data) - self.window_size):
            x_data.append(oh_data[i : i + self.window_size])
            y_data.append(oh_data[i + self.window_size])
        return np.array(x_data), np.array(y_data)

    def train_model(self, oh_data, epochs=30):
        """모델 생성 및 학습"""
        X, y = self.create_dataset(oh_data)
        
        self.model = Sequential()
        self.model.add(LSTM(64, activation='relu', input_shape=(self.window_size, 45)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(45, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # 학습 (verbose=0으로 로그 숨김)
        self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
        return True

    def predict_numbers(self, last_data, past_combinations, count=5):
        """번호 추천 로직"""
        if self.model is None:
            raise Exception("모델이 학습되지 않았습니다.")

        prediction = self.model.predict(last_data.reshape(1, self.window_size, 45), verbose=0)[0]
        prob_norm = prediction / np.sum(prediction) # 확률 정규화
        
        recommendations = []
        while len(recommendations) < count:
            rec_nums = np.random.choice(range(1, 46), size=6, replace=False, p=prob_norm)
            rec_nums.sort()
            
            # 필터링: 과거 이력 및 현재 추천 목록 중복 체크
            combo_tuple = tuple(rec_nums)
            if (combo_tuple not in past_combinations) and (combo_tuple not in recommendations):
                recommendations.append(rec_nums.tolist())
        
        return recommendations