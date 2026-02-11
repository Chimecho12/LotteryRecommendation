import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LottoAI:
    def __init__(self):
        self.model = None
        self.window_size = 5

    def create_dataset(self, oh_data):
        x_data, y_data = [], []
        for i in range(len(oh_data) - self.window_size):
            x_data.append(oh_data[i : i + self.window_size])
            y_data.append(oh_data[i + self.window_size])
        return np.array(x_data), np.array(y_data)

    def train_model(self, oh_data, epochs=30):
        X, y = self.create_dataset(oh_data)
        self.model = Sequential()
        self.model.add(LSTM(64, activation='relu', input_shape=(self.window_size, 45)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(45, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
        return True

    def predict_numbers(self, last_data, past_combinations, count=5, fixed_numbers=None):
        """
        count: 생성할 게임 수
        fixed_numbers: 반드시 포함할 숫자 리스트 (예: [1, 10])
        """
        if self.model is None:
            raise Exception("모델이 학습되지 않았습니다.")
        
        if fixed_numbers is None:
            fixed_numbers = []

        # 고정수가 6개 이상이면 예측할 필요 없음
        if len(fixed_numbers) >= 6:
            return [sorted(fixed_numbers[:6])] * count

        # 1. AI 예측 (기본 확률 분포 생성)
        prediction = self.model.predict(last_data.reshape(1, self.window_size, 45), verbose=0)[0]
        
        # 2. 고정수 제외 처리 (이미 뽑힌 숫자는 확률 0으로 만듦)
        # 고정수에 해당하는 인덱스(번호-1)의 확률을 0으로 설정하여 중복 추출 방지
        for num in fixed_numbers:
            prediction[num-1] = 0 
            
        # 남은 확률 재정규화 (합이 1이 되도록)
        prob_norm = prediction / np.sum(prediction)
        
        recommendations = []
        
        while len(recommendations) < count:
            # 남은 개수만큼만 AI가 뽑음 (6개 - 고정수 개수)
            needed_count = 6 - len(fixed_numbers)
            
            picked_nums = np.random.choice(range(1, 46), size=needed_count, replace=False, p=prob_norm)
            
            # 고정수와 합치기
            final_nums = sorted(fixed_numbers + list(picked_nums))
            
            # 필터링: 과거 이력 및 현재 추천 목록 중복 체크
            combo_tuple = tuple(final_nums)
            if (combo_tuple not in past_combinations) and (combo_tuple not in [tuple(r) for r in recommendations]):
                recommendations.append(final_nums)
        
        return recommendations