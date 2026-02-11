import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self):
        self.df = None
        self.oh_data = None
        self.cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']

    def load_file(self, file_path):
        """엑셀 파일을 읽고 고급 분석을 수행합니다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일을 찾을 수 없습니다.")
        
        self.df = pd.read_excel(file_path)
        
        # 로드 직후 바로 분석 실행
        self._analyze_features()
        return self.df

    def _analyze_features(self):
        """[핵심] 총합, 홀짝, 고저, AC값 등 파생 변수 생성"""
        if self.df is None: return

        print(">>> 고급 분석 지표 계산 중...")

        # 1. 총합 (Sum)
        self.df['총합'] = self.df[self.cols].sum(axis=1)

        # 2. 홀짝 비율 (홀수 개수)
        # (홀수는 2로 나눈 나머지가 1임)
        self.df['홀수개수'] = self.df[self.cols].apply(lambda x: sum(n % 2 for n in x), axis=1)
        self.df['짝수개수'] = 6 - self.df['홀수개수']
        self.df['홀짝비율'] = self.df['홀수개수'].astype(str) + ":" + self.df['짝수개수'].astype(str)

        # 3. 고저 비율 (Low: 1~22, High: 23~45)
        self.df['고숫자개수'] = self.df[self.cols].apply(lambda x: sum(1 for n in x if n >= 23), axis=1)
        self.df['저숫자개수'] = 6 - self.df['고숫자개수']
        self.df['고저비율'] = self.df['저숫자개수'].astype(str) + ":" + self.df['고숫자개수'].astype(str)

        # 4. 끝수 합 (각 번호의 일의 자리 합)
        self.df['끝수합'] = self.df[self.cols].apply(lambda x: sum(n % 10 for n in x), axis=1)

        # 5. AC값 (Arithmetic Complexity) 계산
        # 숫자 간의 차이가 얼마나 다양한지 측정 (높을수록 무작위성 강함)
        def get_ac_value(row):
            nums = sorted(list(row))
            diffs = set()
            for i in range(6):
                for j in range(i + 1, 6):
                    diffs.add(nums[j] - nums[i])
            # AC = (고유한 차이값의 개수) - (번호개수 6 - 1)
            return len(diffs) - 5

        self.df['AC값'] = self.df[self.cols].apply(get_ac_value, axis=1)
        
        print(">>> 분석 완료 (컬럼 추가됨)")

    def preprocess(self):
        """AI 학습용 One-Hot Encoding"""
        if self.df is None: raise ValueError("데이터 없음")
        
        numbers = self.df[self.cols].values
        oh_list = []
        for row in numbers:
            oh_vec = np.zeros(45)
            for num in row:
                oh_vec[int(num)-1] = 1
            oh_list.append(oh_vec)
        
        self.oh_data = np.array(oh_list)
        return self.oh_data

    def get_past_combinations(self):
        """과거 당첨 조합 Set 반환"""
        past_combinations = set()
        for _, row in self.df[self.cols].iterrows():
            past_combinations.add(tuple(sorted(row)))
        return past_combinations