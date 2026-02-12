# src/ai_engine.py
import numpy as np
import pandas as pd  # <--- [수정] 이 부분이 빠져있어서 에러가 났습니다.
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional

class LottoAI:
    def __init__(self):
        self.model = None
        # 분석 기간: 최근 50회(약 1년)의 흐름을 '입력'으로 사용
        self.window_size = 50 
        self.mode = "lotto"
        
        # 확률 테이블
        self.global_lotto_probs = None
        self.global_pension_probs = None
        
        # [NEW] 연금복권 중복 패턴 비율 (예: {6: 0.16, 5: 0.42 ...})
        self.pension_unique_dist = None

    def create_dataset(self, data):
        """시계열 데이터셋 생성"""
        x, y = [], []
        if len(data) <= self.window_size:
            return np.array(x), np.array(y)
            
        for i in range(len(data) - self.window_size):
            x.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(x), np.array(y)

    def train_model(self, data, mode="lotto", epochs=100):
        self.mode = mode
        
        if mode == "lotto":
            self._calculate_global_lotto_probs(data)
        else:
            self._calculate_global_pension_probs(data)
            # [중요] 중복 패턴 비율 계산
            self._calculate_pension_unique_dist(data)

        X, y = self.create_dataset(data)
        
        # 데이터가 부족할 경우 윈도우 사이즈 자동 조정
        if len(X) == 0:
            temp_window = max(1, len(data) - 2)
            print(f"[알림] 데이터가 부족하여 분석 윈도우를 {temp_window}로 조정합니다.")
            self.window_size = temp_window
            X, y = self.create_dataset(data)
            if len(X) == 0: return False

        if mode == "lotto":
            # === 로또 모델 ===
            inputs = Input(shape=(self.window_size, 45))
            x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
            x = Dropout(0.3)(x)
            x = LSTM(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            outputs = Dense(45, activation='sigmoid')(x)
            
            self.model = Model(inputs, outputs)
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
            
        else: 
            # === 연금복권 모델 ===
            inputs = Input(shape=(self.window_size, 7))
            x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
            x = Dropout(0.3)(x)
            x = LSTM(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(7, activation='sigmoid')(x) 
            
            self.model = Model(inputs, outputs)
            self.model.compile(optimizer='adam', loss='mse')

        self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
        return True

    # ========================================================
    # [거시적 분석] 확률 계산 함수들
    # ========================================================
    def _calculate_global_lotto_probs(self, data):
        total_counts = np.sum(data, axis=0)
        self.global_lotto_probs = (total_counts + 1) / np.sum(total_counts + 1)

    def _calculate_global_pension_probs(self, data):
        raw_data = data.copy()
        raw_data[:, 0] = np.round(raw_data[:, 0] * 5.0)
        raw_data[:, 1:] = np.round(raw_data[:, 1:] * 9.0)
        
        self.global_pension_probs = []
        # 조
        jo_counts = np.zeros(6)
        for val in raw_data[:, 0]:
            idx = int(np.clip(val, 1, 5))
            jo_counts[idx] += 1
        self.global_pension_probs.append((jo_counts + 1) / np.sum(jo_counts + 1))
        # 숫자
        for col in range(1, 7):
            num_counts = np.zeros(10)
            for val in raw_data[:, col]:
                idx = int(np.clip(val, 0, 9))
                num_counts[idx] += 1
            self.global_pension_probs.append((num_counts + 1) / np.sum(num_counts + 1))

    def _calculate_pension_unique_dist(self, data):
        """[NEW] 연금복권 6자리 숫자의 '고유 숫자 개수' 분포 계산"""
        raw_data = data.copy()
        # 숫자 6자리만 추출 (인덱스 1~6)
        # 데이터가 정규화되어 있으므로 다시 9.0을 곱해 정수로 복원
        nums = np.round(raw_data[:, 1:] * 9.0).astype(int)
        
        unique_counts = []
        for row in nums:
            unique_counts.append(len(set(row)))
            
        # 분포 계산 (예: {4: 0.33, 5: 0.42, 6: 0.16 ...})
        # 여기서 pd.Series를 사용하므로 import pandas as pd가 필수입니다.
        counts = pd.Series(unique_counts).value_counts(normalize=True).sort_index()
        self.pension_unique_dist = counts.to_dict()

    # ========================================================
    # [로또 전용] 예측 및 필터링
    # ========================================================
    def _calculate_ac(self, nums):
        nums = sorted(list(nums))
        diffs = set(nums[j] - nums[i] for i in range(6) for j in range(i+1, 6))
        return len(diffs) - 5

    def _check_lotto_conditions(self, nums):
        if not (100 <= sum(nums) <= 175): return False
        odd = sum(1 for n in nums if n % 2 == 1)
        if odd == 0 or odd == 6: return False
        high = sum(1 for n in nums if n >= 23)
        if high == 0 or high == 6: return False
        if self._calculate_ac(nums) < 7: return False
        if len(set(n % 10 for n in nums)) == 6: return False
        return True

    def predict_lotto(self, last_data, past_combinations, count=5, fixed_numbers=None):
        if self.model is None: raise Exception("모델 없음")
        if fixed_numbers is None: fixed_numbers = []
        
        lstm_pred = self.model.predict(last_data.reshape(1, self.window_size, 45), verbose=0)[0]
        global_prob = self.global_lotto_probs
        # 하이브리드 결합
        final_prob = (lstm_pred * 0.7) + (global_prob * 0.3)
        
        for n in fixed_numbers: final_prob[n-1] = 0
        final_prob /= np.sum(final_prob)
        
        results = []
        attempts = 0
        while len(results) < count:
            attempts += 1
            needed = 6 - len(fixed_numbers)
            
            # 안전장치
            if attempts > 5000:
                picks = np.random.choice(range(1, 46), size=needed, replace=False, p=final_prob)
                final = sorted(fixed_numbers + list(picks))
                if tuple(final) not in past_combinations: results.append(final)
                continue

            picks = np.random.choice(range(1, 46), size=needed, replace=False, p=final_prob)
            final = sorted(fixed_numbers + list(picks))
            
            if tuple(final) in past_combinations: continue
            if tuple(final) in [tuple(r) for r in results]: continue
            if not self._check_lotto_conditions(final): continue
            
            results.append(final)
        return results

    # ========================================================
    # [연금복권 전용] 예측 로직
    # ========================================================
    def predict_pension(self, last_data, count=5):
        """연금복권 예측: 중복 패턴 확률 반영"""
        if self.model is None or self.pension_unique_dist is None: 
            raise Exception("모델 준비 안됨")
        
        # 1. 트렌드 예측
        ai_trend = self.model.predict(last_data.reshape(1, self.window_size, 7), verbose=0)[0]
        
        results = []
        
        # 목표 중복 패턴 확률분포 (Key: 고유개수, Value: 확률)
        target_keys = list(self.pension_unique_dist.keys())
        target_probs = list(self.pension_unique_dist.values())
        
        for _ in range(count):
            # [NEW] 이번 게임의 목표 '고유 숫자 개수' 설정
            target_unique_count = np.random.choice(target_keys, p=target_probs)
            
            row = []
            
            # 조 예측
            jo_probs = self.global_pension_probs[0][1:]
            ai_jo_weight = np.exp(-0.5 * (np.arange(1, 6) - (ai_trend[0]*5.0))**2)
            final_jo_prob = jo_probs * ai_jo_weight
            final_jo_prob /= np.sum(final_jo_prob)
            row.append(np.random.choice(range(1, 6), p=final_jo_prob))
            
            # 숫자 6자리 예측 (목표 중복 개수 맞출 때까지 재시도)
            best_nums = []
            for _ in range(200): # 시도 횟수 약간 증가
                temp_nums = []
                for i in range(1, 7):
                    hist_probs = self.global_pension_probs[i]
                    ai_val = ai_trend[i] * 9.0
                    ai_weight = np.exp(-0.5 * (np.arange(10) - ai_val)**2)
                    combined_prob = hist_probs * ai_weight
                    combined_prob /= np.sum(combined_prob)
                    temp_nums.append(np.random.choice(range(10), p=combined_prob))
                
                # 생성된 번호의 고유 개수 확인
                if len(set(temp_nums)) == target_unique_count:
                    best_nums = temp_nums
                    break
            
            # 실패 시 그냥 마지막 생성값 사용
            if not best_nums: best_nums = temp_nums
            
            row.extend(best_nums)
            results.append(row)
            
        return results