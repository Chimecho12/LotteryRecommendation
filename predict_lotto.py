import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. 데이터 로드 및 전처리 (Preprocessing)
# ==========================================
def load_and_preprocess_data(file_path):
    print("데이터를 불러오고 전처리를 시작합니다...")
    # 엑셀 파일 읽기
    df = pd.read_excel(file_path)
    
    # 필요한 번호 컬럼만 선택 (번호1 ~ 번호6)
    numbers = df[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values
    
    # [One-Hot Encoding]
    # 로또 번호는 1~45이지만, 인덱스는 0부터 시작하므로 45개 공간을 만듦 (0은 사용 안 함)
    oh_encoder = OneHotEncoder(categories=[range(1, 46)] * 6, sparse_output=False)
    # 하지만 우리는 6개의 독립된 숫자가 아니라 '45개 중 어떤 숫자가 켜졌는지'를 봐야 함
    
    # 각 회차를 45자리 길이의 0과 1 벡터로 변환하는 커스텀 함수
    def numbers_to_oh(rows):
        oh_list = []
        for row in rows:
            # 45개의 0으로 된 배열 생성
            oh_vec = np.zeros(45)
            for num in row:
                # 번호에 해당하는 인덱스(번호-1)를 1로 설정
                oh_vec[int(num)-1] = 1
            oh_list.append(oh_vec)
        return np.array(oh_list)

    oh_data = numbers_to_oh(numbers)
    
    return df, oh_data

# ==========================================
# 2. 시계열 데이터셋 생성 (Dataset Creation)
# ==========================================
def create_dataset(data, window_size=5):
    """
    과거 window_size(예: 5회)만큼의 데이터를 보고
    다음 1회차를 예측하도록 데이터셋(X, y) 분리
    """
    x_data, y_data = [], []
    for i in range(len(data) - window_size):
        x_data.append(data[i : i + window_size]) # 입력: 과거 5주치 데이터
        y_data.append(data[i + window_size])     # 정답: 바로 다음 주 데이터
    return np.array(x_data), np.array(y_data)

# ==========================================
# 3. LSTM 모델 구축 (Model Architecture)
# ==========================================
def build_lstm_model(window_size, feature_num):
    model = Sequential()
    
    # LSTM Layer 1: 패턴 기억
    model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(window_size, feature_num)))
    model.add(Dropout(0.2)) # 과적합 방지
    
    # LSTM Layer 2: 더 깊은 패턴 학습
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output Layer: 45개 번호 각각에 대한 확률 출력 (Sigmoid 사용)
    # Softmax가 아닌 Sigmoid를 쓰는 이유: 로또는 번호가 6개이므로 다중 레이블(Multi-label) 문제임
    model.add(Dense(feature_num, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 4. 번호 추천 및 필터링 (Prediction & Filter)
# ==========================================
def generate_lotto_numbers(model, last_data, past_combinations):
    """
    AI 확률 기반으로 번호를 생성하되, 과거 당첨 내역(past_combinations)에 있으면 제외
    """
    # 다음 회차 확률 예측 (입력 데이터 형태 맞추기)
    prediction = model.predict(last_data.reshape(1, 5, 45), verbose=0)[0]
    
    # 예측된 확률값들을 기반으로 '확률적'으로 번호 추출 (Monte Carlo 방식)
    # 단순히 확률이 높은 순서대로 6개를 뽑으면 매번 똑같은 번호가 나오므로
    # 확률 분포에 따라 랜덤하게 뽑되, 확률이 높은 번호가 더 잘 뽑히게 함
    
    # 확률값들의 합이 1이 되도록 정규화 (np.random.choice를 위해)
    prob_norm = prediction / np.sum(prediction)
    
    while True:
        # 1~45 번호 중 6개를 비복원 추출 (확률 가중치 적용)
        recommended_nums = np.random.choice(range(1, 46), size=6, replace=False, p=prob_norm)
        recommended_nums.sort() # 정렬
        
        # 튜플로 변환 (비교를 위해)
        combo_tuple = tuple(recommended_nums)
        
        # [필터링 로직] 과거 1등 당첨 내역에 있는지 확인
        if combo_tuple not in past_combinations:
            return recommended_nums # 중복 아니면 반환
        else:
            print(f"-> 생성된 조합 {recommended_nums}은 과거 당첨 이력이 있어 제외합니다.")

# ==========================================
# 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 설정
    WINDOW_SIZE = 5   # 과거 5회를 보고 예측
    FILE_PATH = "lotto.xlsx"
    
    # 1. 데이터 로드
    df, oh_data = load_and_preprocess_data(FILE_PATH)
    
    # 과거 당첨 내역 세트(Set) 만들기 (빠른 검색용)
    past_combinations = set()
    for idx, row in df.iterrows():
        nums = sorted([row['번호1'], row['번호2'], row['번호3'], row['번호4'], row['번호5'], row['번호6']])
        past_combinations.add(tuple(nums))
    print(f"-> 역대 {len(past_combinations)}개의 당첨 조합을 필터링 목록에 등록했습니다.")

    # 2. 데이터셋 생성
    X, y = create_dataset(oh_data, WINDOW_SIZE)
    
    # 3. 모델 학습
    print("\n[AI 모델 학습 시작] 잠시만 기다려주세요...")
    model = build_lstm_model(WINDOW_SIZE, 45)
    # epochs=100 정도는 돌려야 패턴을 잡음 (테스트용으로 20만 설정해도 됨)
    model.fit(X, y, epochs=50, batch_size=16, verbose=1)
    
    print("\n[학습 완료] 다음 회차 번호를 생성합니다...")
    
    # 4. 예측을 위한 최근 데이터 준비 (마지막 5주치)
    last_5_weeks = oh_data[-WINDOW_SIZE:]
    
    # 5. 번호 생성 (5게임 추천)
    print("\n" + "="*40)
    print("🔮 AI(LSTM) 기반 추천 번호 (과거 당첨 제외)")
    print("="*40)
    
    for i in range(5):
        nums = generate_lotto_numbers(model, last_5_weeks, past_combinations)
        print(f"게임 {i+1}: {nums} (합계: {sum(nums)})")