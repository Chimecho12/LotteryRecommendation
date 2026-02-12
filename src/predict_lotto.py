import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model # load_model 추가
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
import time
import os

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, update_fn=None):
        self.total_epochs = total_epochs
        self.update_fn = update_fn
        self.start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        if self.update_fn:
            current = epoch + 1
            progress = current / self.total_epochs
            elapsed = time.time() - self.start_time
            avg_time = elapsed / current
            eta = (self.total_epochs - current) * avg_time
            self.update_fn(progress, f"딥러닝 학습 중... [{current}/{self.total_epochs}] (ETA: {int(eta)}s)")

class LottoAI:
    def __init__(self):
        self.model = None
        self.window_size = 50 
        self.mode = "lotto"
        self.global_lotto_probs = None
        self.global_pension_probs = None
        self.pension_unique_dist = None
        # 모델 저장 폴더 생성
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")

    def create_dataset(self, data):
        x, y = [], []
        if len(data) <= self.window_size: return np.array(x), np.array(y)
        for i in range(len(data) - self.window_size):
            x.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(x), np.array(y)

    def train_model(self, data, mode="lotto", epochs=100, progress_cb=None, file_path=None):
        """
        [수정] file_path: 원본 파일 경로 (캐시 검증용)
        """
        self.mode = mode
        
        # 1. 통계 계산
        if progress_cb: progress_cb(0.05, "데이터 통계 분석 중...")
        if mode == "lotto":
            self._calculate_global_lotto_probs(data)
        else:
            self._calculate_global_pension_probs(data)
            self._calculate_pension_unique_dist(data)

        # 2. 모델 캐시 확인
        model_filename = f"saved_models/model_{mode}.keras"
        should_retrain = True
        
        if file_path and os.path.exists(model_filename):
            file_mtime = os.path.getmtime(file_path)
            model_mtime = os.path.getmtime(model_filename)
            
            # 원본 파일보다 모델 파일이 더 최신인 경우 로드
            if model_mtime > file_mtime:
                try:
                    if progress_cb: progress_cb(0.1, "저장된 분석 모델 불러오는 중...")
                    self.model = load_model(model_filename)
                    print(f"[시스템] 캐시된 모델을 로드했습니다: {model_filename}")
                    should_retrain = False
                    time.sleep(0.5) # 사용자가 로딩 메시지를 볼 수 있게 잠깐 대기
                except Exception as e:
                    print(f"[오류] 모델 로드 실패, 재학습합니다: {e}")
        
        if not should_retrain:
            return True

        # 3. 모델 재학습 (캐시가 없거나 파일이 변경된 경우)
        X, y = self.create_dataset(data)
        if len(X) == 0:
            self.window_size = max(1, len(data) - 2)
            X, y = self.create_dataset(data)
            if len(X) == 0: return False

        if progress_cb: progress_cb(0.1, "새로운 데이터로 학습 시작...")
        
        # 모델 구조 정의
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

        callbacks = []
        if progress_cb:
            callbacks.append(TrainingCallback(epochs, progress_cb))

        self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=0, callbacks=callbacks)
        
        # 학습 완료 후 모델 저장
        self.model.save(model_filename)
        print(f"[시스템] 학습된 모델을 저장했습니다: {model_filename}")
        
        return True

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

    def _get_lotto_reason(self, nums, final_prob):
        reasons = []
        s = sum(nums)
        if 120 <= s <= 150: reasons.append("[빈도가 높은 합계 구간]")
        elif s < 120: reasons.append("[낮은 수 위주]")
        else: reasons.append("[높은 수 위주]")
        top_indices = np.argsort(final_prob)[::-1][:15]
        ai_match_count = sum(1 for n in nums if (n-1) in top_indices)
        if ai_match_count >= 3: reasons.append(f"[분석에 적합한 추천] {ai_match_count}개 포함")
        odd = sum(1 for n in nums if n % 2 == 1)
        if odd == 3: reasons.append("[홀짝 밸런스]")
        return ", ".join(reasons)

    def predict_lotto(self, last_data, past_combinations, count=5, fixed_numbers=None, progress_cb=None):
        if self.model is None: raise Exception("모델 없음")
        if fixed_numbers is None: fixed_numbers = []
        if progress_cb: progress_cb(0.9, "번호 조합 생성 중...")
        lstm_pred = self.model.predict(last_data.reshape(1, self.window_size, 45), verbose=0)[0]
        global_prob = self.global_lotto_probs
        final_prob = (lstm_pred * 0.7) + (global_prob * 0.3)
        for n in fixed_numbers: final_prob[n-1] = 0
        final_prob /= np.sum(final_prob)
        results = []
        attempts = 0
        while len(results) < count:
            attempts += 1
            if progress_cb and attempts % 100 == 0:
                current_percent = 0.9 + (0.1 * (len(results) / count))
                progress_cb(current_percent, f"필터링 중... ({len(results)}/{count})")
            needed = 6 - len(fixed_numbers)
            if attempts > 5000:
                picks = np.random.choice(range(1, 46), size=needed, replace=False, p=final_prob)
                final = sorted(fixed_numbers + list(picks))
                if tuple(final) not in past_combinations:
                    reason = "⚠️필터 조건 완화"
                    results.append((final, reason))
                continue
            picks = np.random.choice(range(1, 46), size=needed, replace=False, p=final_prob)
            final = sorted(fixed_numbers + list(picks))
            if tuple(final) in past_combinations: continue
            if tuple(final) in [tuple(r[0]) for r in results]: continue
            if not self._check_lotto_conditions(final): continue
            reason = self._get_lotto_reason(final, final_prob)
            results.append((final, reason))
        if progress_cb: progress_cb(1.0, "완료!")
        return results

    def _get_pension_reason(self, row, target_unique_count, ai_trend):
        reasons = []
        jo = row[0]
        predicted_jo = (ai_trend[0] * 5.0)
        if abs(jo - predicted_jo) < 1.0: reasons.append("[예측 조]")
        if target_unique_count == 6: reasons.append("[모두 다른 숫자]")
        elif target_unique_count == 5: reasons.append("[1쌍 중복 패턴(42%확률)]")
        elif target_unique_count == 4: reasons.append("[2쌍 중복 패턴(33%확률)]")
        nums = row[1:]
        high_nums = sum(1 for n in nums if n >= 5)
        if high_nums >= 4: reasons.append("[높은 수 위주]")
        elif high_nums <= 2: reasons.append("[낮은 수 위주]")
        return ", ".join(reasons)

    def predict_pension(self, last_data, count=5, progress_cb=None):
        if self.model is None or self.pension_unique_dist is None: 
            raise Exception("모델 준비 안됨")
        if progress_cb: progress_cb(0.9, "연금복권 번호 조합 중...")
        ai_trend = self.model.predict(last_data.reshape(1, self.window_size, 7), verbose=0)[0]
        results = []
        target_keys = list(self.pension_unique_dist.keys())
        target_probs = list(self.pension_unique_dist.values())
        for idx in range(count):
            if progress_cb:
                progress_cb(0.9 + (0.1 * ((idx + 1) / count)), f"번호 생성 중... ({idx+1}/{count})")
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
            reason = self._get_pension_reason(row, target_unique_count, ai_trend)
            results.append((row, reason))
        if progress_cb: progress_cb(1.0, "완료!")
        return results