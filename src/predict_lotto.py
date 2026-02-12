# src/ai_engine.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
import time

# [NEW] 학습 진행 상황을 GUI로 실시간 전달하는 콜백 클래스
class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, update_fn=None):
        self.total_epochs = total_epochs
        self.update_fn = update_fn
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if self.update_fn:
            current = epoch + 1
            progress = current / self.total_epochs
            
            # 경과 시간 및 남은 시간(ETA) 계산
            elapsed = time.time() - self.start_time
            avg_time_per_epoch = elapsed / current
            remaining_epochs = self.total_epochs - current
            eta = remaining_epochs * avg_time_per_epoch
            
            # GUI로 정보 전달 (진행률 0.0~1.0, 메시지)
            msg = f"딥러닝 학습 중... [{current}/{self.total_epochs}] (남은 시간: 약 {int(eta)}초)"
            self.update_fn(progress, msg)

class LottoAI:
    def __init__(self):
        self.model = None
        self.window_size = 50 
        self.mode = "lotto"
        self.global_lotto_probs = None
        self.global_pension_probs = None
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

    def train_model(self, data, mode="lotto", epochs=100, progress_cb=None):
        """
        [수정] progress_cb: 진행률을 업데이트할 GUI 함수
        """
        self.mode = mode
        
        # 1. 확률 계산 단계 (빠르지만 사용자에게 알림)
        if progress_cb: progress_cb(0.05, "데이터 통계 분석 중...")
        
        if mode == "lotto":
            self._calculate_global_lotto_probs(data)
        else:
            self._calculate_global_pension_probs(data)
            self._calculate_pension_unique_dist(data)

        # 2. 데이터셋 생성
        X, y = self.create_dataset(data)
        
        if len(X) == 0:
            self.window_size = max(1, len(data) - 2)
            X, y = self.create_dataset(data)
            if len(X) == 0: return False

        # 3. 모델 구축
        if progress_cb: progress_cb(0.1, "AI 모델 구조 설계 중...")
        
        if mode == "lotto":
            inputs = Input(shape=(self.window_size, 45))
            x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
            x = Dropout(0.3)(x)
            x = LSTM(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            outputs = Dense(45, activation='sigmoid')(x)
            self.model = Model(inputs, outputs)
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
        else: 
            inputs = Input(shape=(self.window_size, 7))
            x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
            x = Dropout(0.3)(x)
            x = LSTM(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(7, activation='sigmoid')(x) 
            self.model = Model(inputs, outputs)
            self.model.compile(optimizer='adam', loss='mse')

        # 4. 학습 실행 (콜백 연결)
        callbacks = []
        if progress_cb:
            # 학습 구간은 전체 진행률의 10% ~ 90%를 담당한다고 가정하고 매핑할 수도 있지만,
            # 여기서는 단순하게 콜백 내에서 메시지를 업데이트하는 방식으로 처리
            callbacks.append(TrainingCallback(epochs, progress_cb))

        self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=0, callbacks=callbacks)
        return True

    # ... (확률 계산 함수들 _calculate_... 생략 / 기존 코드 유지) ...
    def _calculate_global_lotto_probs(self, data):
        total_counts = np.sum(data, axis=0)
        self.global_lotto_probs = (total_counts + 1) / np.sum(total_counts + 1)

    def _calculate_global_pension_probs(self, data):
        raw_data = data.copy()
        raw_data[:, 0] = np.round(raw_data[:, 0] * 5.0)
        raw_data[:, 1:] = np.round(raw_data[:, 1:] * 9.0)
        
        self.global_pension_probs = []
        jo_counts = np.zeros(6)
        for val in raw_data[:, 0]:
            idx = int(np.clip(val, 1, 5))
            jo_counts[idx] += 1
        self.global_pension_probs.append((jo_counts + 1) / np.sum(jo_counts + 1))
        
        for col in range(1, 7):
            num_counts = np.zeros(10)
            for val in raw_data[:, col]:
                idx = int(np.clip(val, 0, 9))
                num_counts[idx] += 1
            self.global_pension_probs.append((num_counts + 1) / np.sum(num_counts + 1))

    def _calculate_pension_unique_dist(self, data):
        raw_data = data.copy()
        nums = np.round(raw_data[:, 1:] * 9.0).astype(int)
        unique_counts = []
        for row in nums: unique_counts.append(len(set(row)))
        counts = pd.Series(unique_counts).value_counts(normalize=True).sort_index()
        self.pension_unique_dist = counts.to_dict()
        
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

    def predict_lotto(self, last_data, past_combinations, count=5, fixed_numbers=None, progress_cb=None):
        if self.model is None: raise Exception("모델 없음")
        if fixed_numbers is None: fixed_numbers = []
        
        # 예측 시작 알림
        if progress_cb: progress_cb(0.9, "번호 조합 생성 및 필터링 중...")
        
        lstm_pred = self.model.predict(last_data.reshape(1, self.window_size, 45), verbose=0)[0]
        global_prob = self.global_lotto_probs
        final_prob = (lstm_pred * 0.7) + (global_prob * 0.3)
        
        for n in fixed_numbers: final_prob[n-1] = 0
        final_prob /= np.sum(final_prob)
        
        results = []
        attempts = 0
        while len(results) < count:
            attempts += 1
            
            # 진행률 업데이트 (생성된 게임 수 기준)
            if progress_cb and attempts % 100 == 0:
                current_percent = 0.9 + (0.1 * (len(results) / count))
                progress_cb(current_percent, f"고급 필터링 적용 중... ({len(results)}/{count} 완료)")

            needed = 6 - len(fixed_numbers)
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
            
        if progress_cb: progress_cb(1.0, "완료!")
        return results

    def predict_pension(self, last_data, count=5, progress_cb=None):
        if self.model is None or self.pension_unique_dist is None: 
            raise Exception("모델 준비 안됨")
            
        if progress_cb: progress_cb(0.9, "연금복권 번호 조합 중...")
        
        ai_trend = self.model.predict(last_data.reshape(1, self.window_size, 7), verbose=0)[0]
        results = []
        target_keys = list(self.pension_unique_dist.keys())
        target_probs = list(self.pension_unique_dist.values())
        
        for idx in range(count):
            # 진행률 업데이트
            if progress_cb:
                current_percent = 0.9 + (0.1 * ((idx + 1) / count))
                progress_cb(current_percent, f"번호 생성 중... ({idx+1}/{count})")

            target_unique_count = np.random.choice(target_keys, p=target_probs)
            row = []
            
            jo_probs = self.global_pension_probs[0][1:]
            ai_jo_weight = np.exp(-0.5 * (np.arange(1, 6) - (ai_trend[0]*5.0))**2)
            final_jo_prob = jo_probs * ai_jo_weight
            final_jo_prob /= np.sum(final_jo_prob)
            row.append(np.random.choice(range(1, 6), p=final_jo_prob))
            
            best_nums = []
            for _ in range(200):
                temp_nums = []
                for i in range(1, 7):
                    hist_probs = self.global_pension_probs[i]
                    ai_val = ai_trend[i] * 9.0
                    ai_weight = np.exp(-0.5 * (np.arange(10) - ai_val)**2)
                    combined_prob = hist_probs * ai_weight
                    combined_prob /= np.sum(combined_prob)
                    temp_nums.append(np.random.choice(range(10), p=combined_prob))
                
                if len(set(temp_nums)) == target_unique_count:
                    best_nums = temp_nums
                    break
            
            if not best_nums: best_nums = temp_nums
            row.extend(best_nums)
            results.append(row)
            
        if progress_cb: progress_cb(1.0, "완료!")
        return results