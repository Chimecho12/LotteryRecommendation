import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 한글 폰트 설정 (깨짐 방지)
import matplotlib.font_manager as fm
import platform

# OS별 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 1. 엑셀 데이터 로드
try:
    df = pd.read_excel("lotto_history.xlsx")
    print("엑셀 파일을 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    exit()

# 번호 데이터만 추출 (컬럼명이 '번호1' ~ '번호6'이라고 가정)
# 만약 컬럼명이 다르다면 아래 리스트를 수정해주세요.
cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
lotto_nums = df[cols]

# ==========================================
# 분석 1: 1~45번 모든 번호의 출현 빈도 분석
# ==========================================
# 모든 번호를 1차원 리스트로 펼침
all_numbers = lotto_nums.values.flatten()
# 1~45번까지 빈도 계산 (나오지 않은 번호가 있어도 0으로 표시하기 위해 range 사용)
count_dict = Counter(all_numbers)
frequency_df = pd.DataFrame({
    '번호': list(range(1, 46)),
    '빈도': [count_dict.get(i, 0) for i in range(1, 46)]
})

# ==========================================
# 분석 2: 당첨 번호 '조합' 상위 20개 분석
# ==========================================
# 각 회차의 번호를 정렬하여 문자열로 만듦 (예: "1, 10, 23, ...")
# 튜플로 변환해야 카운트가 가능함
df['조합'] = lotto_nums.apply(lambda row: tuple(sorted(row)), axis=1)
combo_counts = df['조합'].value_counts().head(20)

# 시각화를 위해 튜플을 보기 좋은 문자열로 변환
combo_labels = [str(c).replace('(', '').replace(')', '') for c in combo_counts.index]
combo_values = combo_counts.values

# ==========================================
# 시각화 그리기
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(14, 14)) # 2행 1열 구조

# [그래프 1] 1~45번 전체 빈도 시각화
sns.barplot(
    x='번호', y='빈도', data=frequency_df, 
    ax=axes[0], palette="viridis", edgecolor='black'
)
axes[0].set_title("역대 로또 번호별 당첨 횟수 (1~45번 전체)", fontsize=16, fontweight='bold')
axes[0].set_xlabel("번호", fontsize=12)
axes[0].set_ylabel("당첨 횟수", fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# 각 막대 위에 숫자 표시
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), 
                     textcoords = 'offset points',
                     fontsize=9)

# [그래프 2] 상위 20개 조합 시각화
sns.barplot(
    x=combo_values, y=combo_labels, 
    ax=axes[1], palette="magma", edgecolor='black'
)
axes[1].set_title("가장 많이 등장한 당첨 번호 조합 Top 20", fontsize=16, fontweight='bold')
axes[1].set_xlabel("당첨 횟수 (중복 등장 횟수)", fontsize=12)
axes[1].set_ylabel("번호 조합", fontsize=12)
axes[1].grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()