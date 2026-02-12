# src/ai_engine.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional

class LottoAI:
    def __init__(self):
        self.model = None
        # 분석 기간: 최근 1년(50회)의 흐름을 '입력'으로 사용
        self.window_size = 50 
        self.mode = "lotto"
        
        # [핵심] 전체 데이터의 흐름을 담을 확률 테이블
        self.global_lotto_probs = None    # 로또용 (1~45번 전체 빈도)
        self.global_pension_probs = None  # 연금복권용 (각 자리별 전체 빈도)

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
        
        # 1. [거시적 분석] 전체 데이터의 통계적 확률 미리 계산
        if mode == "lotto":
            self._calculate_global_lotto_probs(data)
        else:
            self._calculate_global_pension_probs(data)

        # 2. [미시적 분석] LSTM 모델 학습 (최근 흐름 파악용)
        X, y = self.create_dataset(data)
        
        # 데이터 부족 시 예외처리
        if len(X) == 0:
            temp_window = max(1, len(data) - 2)
            print(f"[알림] 데이터가 부족하여 분석 윈도우를 {temp_window}로 조정합니다.")
            self.window_size = temp_window
            X, y = self.create_dataset(data)
            if len(X) == 0: return False

        if mode == "lotto":
            # === 로또 모델 (양방향 LSTM) ===
            inputs = Input(shape=(self.window_size, 45))
            x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
            x = Dropout(0.3)(x)
            x = LSTM(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            outputs = Dense(45, activation='sigmoid')(x)
            
            self.model = Model(inputs, outputs)
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
            
        else: 
            # === 연금복권 모델 (양방향 LSTM) ===
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
    # [거시적 분석] 전체 데이터 확률 계산 함수들
    # ========================================================
    def _calculate_global_lotto_probs(self, data):
        """로또: 1~45번 번호가 역대 몇 번 나왔는지 전체 확률 계산"""
        # data shape: (N, 45) One-Hot 벡터
        # 열(Column)별로 합치면 번호별 총 등장 횟수가 됨
        total_counts = np.sum(data, axis=0) # [1번횟수, 2번횟수, ..., 45번횟수]
        
        # 확률로 변환 (Laplace Smoothing +1)
        self.global_lotto_probs = (total_counts + 1) / np.sum(total_counts + 1)

    def _calculate_global_pension_probs(self, data):
        """연금복권: 각 자리별(조~6번째) 숫자 빈도 전체 확률 계산"""
        # 데이터 복원 (정규화 해제)
        raw_data = data.copy()
        raw_data[:, 0] = np.round(raw_data[:, 0] * 5.0) # 조
        raw_data[:, 1:] = np.round(raw_data[:, 1:] * 9.0) # 숫자
        
        self.global_pension_probs = []
        
        # 1. 조 (Group) 확률
        jo_counts = np.zeros(6)
        for val in raw_data[:, 0]:
            idx = int(np.clip(val, 1, 5))
            jo_counts[idx] += 1
        self.global_pension_probs.append((jo_counts + 1) / np.sum(jo_counts + 1))
        
        # 2. 숫자 (0~9) 확률
        for col in range(1, 7):
            num_counts = np.zeros(10)
            for val in raw_data[:, col]:
                idx = int(np.clip(val, 0, 9))
                num_counts[idx] += 1
            self.global_pension_probs.append((num_counts + 1) / np.sum(num_counts + 1))

    # ========================================================
    # [예측 및 필터링]
    # ========================================================
    def _calculate_ac(self, nums):
        nums = sorted(list(nums))
        diffs = set(nums[j] - nums[i] for i in range(6) for j in range(i+1, 6))
        return len(diffs) - 5

    def _check_lotto_conditions(self, nums):
        # 통계적 필터 (기존 동일)
        if not (100 <= sum(nums) <= 175): return False
        odd = sum(1 for n in nums if n % 2 == 1)
        if odd == 0 or odd == 6: return False
        high = sum(1 for n in nums if n >= 23)
        if high == 0 or high == 6: return False
        if self._calculate_ac(nums) < 7: return False
        if len(set(n % 10 for n in nums)) == 6: return False
        return True

    def predict_lotto(self, last_data, past_combinations, count=5, fixed_numbers=None):
        """
        [로또 예측] 전체 역사(Global) + 최근 추세(LSTM) 결합
        """
        if self.model is None: raise Exception("모델 없음")
        if fixed_numbers is None: fixed_numbers = []
        
        # 1. LSTM 예측 (최근 50회 흐름 반영)
        lstm_pred = self.model.predict(last_data.reshape(1, self.window_size, 45), verbose=0)[0]
        
        # 2. 전체 통계 확률 (전체 역사 반영)
        global_prob = self.global_lotto_probs
        
        # 3. [핵심] 두 확률의 결합 (가중치 적용)
        # LSTM(트렌드) 70% + Global(역사) 30% 비중으로 결합
        final_prob = (lstm_pred * 0.7) + (global_prob * 0.3)
        
        # 고정수 처리
        for n in fixed_numbers: final_prob[n-1] = 0
        final_prob /= np.sum(final_prob) # 정규화
        
        results = []
        attempts = 0
        
        while len(results) < count:
            attempts += 1
            if attempts > 5000:
                # 안전장치
                needed = 6 - len(fixed_numbers)
                picks = np.random.choice(range(1, 46), size=needed, replace=False, p=final_prob)
                final = sorted(fixed_numbers + list(picks))
                if tuple(final) not in past_combinations: results.append(final)
                continue

            needed = 6 - len(fixed_numbers)
            picks = np.random.choice(range(1, 46), size=needed, replace=False, p=final_prob)
            final = sorted(fixed_numbers + list(picks))
            
            if tuple(final) in past_combinations: continue
            if tuple(final) in [tuple(r) for r in results]: continue
            if not self._check_lotto_conditions(final): continue
            
            results.append(final)
        return results

    def predict_pension(self, last_data, count=5):
        """
        [연금복권 예측] 전체 역사(Global) + 최근 추세(LSTM) 결합
        """
        if self.model is None or self.global_pension_probs is None: 
            raise Exception("모델 또는 확률 테이블 없음")
        
        # 1. LSTM 트렌드 예측 (0.0 ~ 1.0 값)
        ai_trend = self.model.predict(last_data.reshape(1, self.window_size, 7), verbose=0)[0]
        
        results = []
        for _ in range(count):
            row = []
            
            # --- 조 (Group) 예측 ---
            # 전체 역사적 빈도
            hist_probs = self.global_pension_probs[0][1:]
            
            # AI 트렌드를 가우시안 분포로 변환하여 가중치 생성
            ai_val = ai_trend[0] * 5.0
            ai_weight = np.exp(-0.5 * (np.arange(1, 6) - ai_val)**2)
            
            # [결합] 역사 * 트렌드
            combined_prob = hist_probs * ai_weight
            combined_prob /= np.sum(combined_prob)
            
            jo = np.random.choice(range(1, 6), p=combined_prob)
            row.append(jo)
            
            # --- 숫자 6자리 예측 ---
            for i in range(1, 7):
                # 전체 역사적 빈도
                hist_probs = self.global_pension_probs[i]
                
                # AI 트렌드
                ai_val = ai_trend[i] * 9.0
                ai_weight = np.exp(-0.5 * (np.arange(10) - ai_val)**2)
                
                # [결합]
                combined_prob = hist_probs * ai_weight
                combined_prob /= np.sum(combined_prob)
                
                num = np.random.choice(range(10), p=combined_prob)
                row.append(num)
            
            results.append(row)
            
        return results