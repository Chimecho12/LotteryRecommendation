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
        x_data, y_data = [], []
        for i in range(len(oh_data) - self.window_size):
            x_data.append(oh_data[i : i + self.window_size])
            y_data.append(oh_data[i + self.window_size])
        return np.array(x_data), np.array(y_data)

    def train_model(self, oh_data, epochs=50):
        """모델 학습 (epochs를 50으로 늘려 정확도 향상)"""
        X, y = self.create_dataset(oh_data)
        
        self.model = Sequential()
        self.model.add(LSTM(128, activation='relu', input_shape=(self.window_size, 45))) # 노드 수 증가
        self.model.add(Dropout(0.3))
        self.model.add(Dense(45, activation='sigmoid'))
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
        return True

    def _calculate_ac(self, nums):
        """AC값(복잡도) 계산 함수"""
        nums = sorted(list(nums))
        diffs = set()
        for i in range(6):
            for j in range(i + 1, 6):
                diffs.add(nums[j] - nums[i])
        # AC = (고유한 차이값 개수) - (6 - 1)
        return len(diffs) - 5

    def _check_stat_conditions(self, nums):
        # 1. 총합 필터 (일반적인 범위 100 ~ 175)
        total_sum = sum(nums)
        if not (100 <= total_sum <= 175):
            return False

        # 2. 홀짝 비율 (Odd/Even Ratio)
        # 6:0(모두 홀수) 또는 0:6(모두 짝수) 제외
        odd_count = sum(1 for n in nums if n % 2 == 1)
        if odd_count == 0 or odd_count == 6:
            return False

        # 3. 고저 비율 (High/Low Ratio)
        # 저(1~22), 고(23~45). 쏠림 현상(6:0, 0:6) 제외
        high_count = sum(1 for n in nums if n >= 23)
        if high_count == 0 or high_count == 6:
            return False

        # 4. AC값 (Arithmetic Complexity)
        # AC값이 7 이상이어야 함 (0~6은 너무 규칙적이라 제외)
        ac_val = self._calculate_ac(nums)
        if ac_val < 7:
            return False

        # 5. 끝수 분석 (End Digit)
        # 동일한 끝수가 하나라도 있어야 함 (확률 80% 이상)
        # 예: 3, 13 (끝수 3으로 동일)
        end_digits = [n % 10 for n in nums]
        if len(set(end_digits)) == 6: # 끝수가 모두 다르면(중복이 없으면) False
            return False

        # 6. 연번 (Consecutive Numbers)
        # 연번은 통계적으로 자주 나오므로(50% 이상), 
        # "연번이 있어서 제외"하는 로직은 넣지 않음 (사용자 요청 반영)
        
        return True

    def predict_numbers(self, last_data, past_combinations, count=5, fixed_numbers=None):
        if self.model is None:
            raise Exception("모델이 학습되지 않았습니다.")
        
        if fixed_numbers is None: fixed_numbers = []
        
        # 고정수가 6개 이상이면 바로 반환
        if len(fixed_numbers) >= 6:
            return [sorted(fixed_numbers[:6])] * count

        # AI 예측 (확률 분포 생성)
        prediction = self.model.predict(last_data.reshape(1, self.window_size, 45), verbose=0)[0]
        
        # 고정수 확률 0 처리 (중복 방지)
        for num in fixed_numbers:
            prediction[num-1] = 0 
            
        prob_norm = prediction / np.sum(prediction)
        
        recommendations = []
        attempts = 0 # 무한 루프 방지용 카운터
        
        while len(recommendations) < count:
            attempts += 1
            # 10만 번 시도해도 조건 만족하는 번호가 안 나오면 그냥 확률 높은 거 줌 (안전장치)
            if attempts > 100000:
                print("조건 만족 실패: 확률 기반 강제 추출")
                needed_count = 6 - len(fixed_numbers)
                picked_nums = np.random.choice(range(1, 46), size=needed_count, replace=False, p=prob_norm)
                final_nums = sorted(fixed_numbers + list(picked_nums))
                recommendations.append(final_nums)
                continue

            # 1. AI 확률 기반 추출
            needed_count = 6 - len(fixed_numbers)
            picked_nums = np.random.choice(range(1, 46), size=needed_count, replace=False, p=prob_norm)
            final_nums = sorted(fixed_numbers + list(picked_nums))
            
            combo_tuple = tuple(final_nums)
            
            # 2. [필터링 1단계] 과거 1등 당첨 이력 제외
            if combo_tuple in past_combinations:
                continue
            
            # 3. [필터링 1단계] 현재 추천 목록 중복 제외
            if combo_tuple in [tuple(r) for r in recommendations]:
                continue
                
            # 4. [필터링 2단계] 통계적 조건 검증 (홀짝, 고저, AC값 등)
            if not self._check_stat_conditions(final_nums):
                continue # 조건 안 맞으면 버리고 다시 뽑기

            # 모든 관문 통과!
            recommendations.append(final_nums)
        
        return recommendations