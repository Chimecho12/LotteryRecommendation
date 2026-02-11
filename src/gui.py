import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import threading
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# 로직 모듈
from src.data_loader import DataLoader
from src.predict_lotto import LottoAI

if platform.system() == 'Windows':
    # 윈도우인 경우 'Malgun Gothic' (맑은 고딕)
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':
    # 맥(Mac)인 경우 'AppleGothic'
    rc('font', family='AppleGothic')
else:
    # 리눅스인 경우 (보통 NanumGothic 설치 필요)
    rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

ctk.set_appearance_mode("Dark") 
ctk.set_default_color_theme("blue")

class LottoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Lotto Predictor Ultimate")
        self.geometry("800x850")
        
        self.loader = DataLoader()
        self.ai = LottoAI()
        self._init_ui()

    # ... (기존 _init_ui, log, load_file, start_thread, run_ai 코드는 동일하므로 생략) ...
    # ... (위의 코드들 복사해서 그대로 쓰시면 됩니다) ...

    def _init_ui(self):
        # (이전 답변의 _init_ui 코드와 동일하게 작성해주세요)
        # 편의를 위해 버튼 연결 부분만 적어드립니다.
        # ...
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # === 1. 헤더 ===
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.header_frame, text="AI 로또 분석기 Pro", font=("Arial", 24, "bold")).pack(pady=10)
        
        self.btn_file = ctk.CTkButton(self.header_frame, text="📂 엑셀 파일 불러오기", command=self.load_file)
        self.btn_file.pack(padx=20, pady=(0,5), fill="x")
        self.lbl_status = ctk.CTkLabel(self.header_frame, text="파일 없음", text_color="gray")
        self.lbl_status.pack(pady=(0,10))

        # === 2. 설정 영역 ===
        self.setting_frame = ctk.CTkFrame(self)
        self.setting_frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        self.setting_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(self.setting_frame, text="생성할 게임 수:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_count = ctk.CTkEntry(self.setting_frame, placeholder_text="예: 5")
        self.entry_count.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.entry_count.insert(0, "5")

        ctk.CTkLabel(self.setting_frame, text="고정수 (쉼표 구분):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_fixed = ctk.CTkEntry(self.setting_frame, placeholder_text="예: 7, 15 (없으면 비움)")
        self.entry_fixed.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # === 3. 실행 버튼 ===
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_analyze = ctk.CTkButton(self.btn_frame, text="📊 종합 분석 리포트", command=self.show_analysis, 
                                         state="disabled", fg_color="#00897B")
        self.btn_analyze.grid(row=0, column=0, padx=(0,5), sticky="ew", ipady=5)

        self.btn_predict = ctk.CTkButton(self.btn_frame, text="🔮 AI 예측 시작", command=self.start_thread, 
                                         state="disabled", fg_color="#3949AB")
        self.btn_predict.grid(row=0, column=1, padx=(5,0), sticky="ew", ipady=5)

        # === 4. 로그 창 ===
        self.log_textbox = ctk.CTkTextbox(self, font=("Consolas", 14))
        self.log_textbox.grid(row=3, column=0, padx=20, pady=20, sticky="nsew")
        self.log_textbox.insert("0.0", "준비 완료.\n")
        self.log_textbox.configure(state="disabled")

    def log(self, msg):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", msg + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            try:
                self.loader.load_file(path)
                self.lbl_status.configure(text=f"로드 완료: {os.path.basename(path)}", text_color="#66BB6A")
                self.btn_analyze.configure(state="normal")
                self.btn_predict.configure(state="normal")
                self.log(f"[시스템] 데이터 로드 성공!")
            except Exception as e:
                self.log(f"[에러] {e}")

    def start_thread(self):
        self.btn_predict.configure(state="disabled", text="학습 중...")
        threading.Thread(target=self.run_ai).start()

    def run_ai(self):
        # (이전 답변과 동일한 AI 실행 로직)
        try:
            try:
                game_count = int(self.entry_count.get())
                if game_count < 1: game_count = 1
            except: game_count = 5

            fixed_nums = []
            fixed_str = self.entry_fixed.get().strip()
            if fixed_str:
                try:
                    fixed_nums = [int(n.strip()) for n in fixed_str.split(',') if n.strip().isdigit()]
                    fixed_nums = [n for n in fixed_nums if 1 <= n <= 45]
                    fixed_nums = sorted(list(set(fixed_nums)))[:5]
                except: pass

            self.log(f"\n>>> 설정: {game_count}게임 / 고정수: {fixed_nums}")
            self.log(">>> 데이터 학습 시작...")
            oh_data = self.loader.preprocess()
            self.ai.train_model(oh_data)
            
            past_combos = self.loader.get_past_combinations()
            last_data = oh_data[-self.ai.window_size:]
            results = self.ai.predict_numbers(last_data, past_combos, count=game_count, fixed_numbers=fixed_nums)
            
            self.log("\n====== AI 추천 번호 ======")
            for i, nums in enumerate(results):
                self.log(f" GAME {i+1}: {nums} (합계: {sum(nums)})")
            self.log("==========================")
        except Exception as e:
            self.log(f"[오류] {e}")
        finally:
            self.btn_predict.configure(state="normal", text="🔮 AI 예측 시작")

    def show_analysis(self):
        # 1. 새 창(Toplevel) 생성
        win = ctk.CTkToplevel(self)
        win.title("종합 분석 리포트")
        win.geometry("950x800")
        
        # 창이 맨 앞으로 오게 설정
        win.attributes('-topmost', True)
        win.after(100, lambda: win.attributes('-topmost', False))

        # 2. 타이틀 레이블
        ctk.CTkLabel(win, text="📊 AI 로또 데이터 분석 리포트", 
                     font=("Arial", 20, "bold")).pack(pady=10)

        # 3. [핵심 변경] 탭 대신 스크롤 가능한 프레임 사용
        # 스마트폰처럼 위아래로 스크롤하며 모든 그래프를 볼 수 있습니다.
        scroll_frame = ctk.CTkScrollableFrame(win, width=900, height=700)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        df = self.loader.df
        plt.style.use('dark_background') # 다크 테마 적용

        # --- 그래프 1: 번호별 빈도 ---
        self._add_report_section(scroll_frame, "1. 번호별 당첨 횟수 분포")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        all_nums = df[['번호1','번호2','번호3','번호4','번호5','번호6']].values.flatten()
        sns.histplot(all_nums, bins=45, ax=ax1, color='#29B6F6', edgecolor='black')
        ax1.set_xlim(0, 46)
        self._embed_graph(fig1, scroll_frame)

        # --- 그래프 2: 총합 분포 ---
        self._add_report_section(scroll_frame, "2. 당첨 번호 합계(Sum) 분포")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        if '총합' in df.columns:
            sns.histplot(df['총합'], kde=True, ax=ax2, color='#FFCA28', bins=30)
            # 평균선 표시
            avg_sum = df['총합'].mean()
            ax2.axvline(avg_sum, color='red', linestyle='--', label=f'평균: {int(avg_sum)}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "데이터 로드 필요", ha='center', color='white')
        self._embed_graph(fig2, scroll_frame)

        # --- 그래프 3: 홀짝 및 고저 비율 ---
        self._add_report_section(scroll_frame, "3. 홀짝(Left) & 고저(Right) 비율")
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(8, 4))
        
        if '홀짝비율' in df.columns:
            oe_counts = df['홀짝비율'].value_counts().head(5)
            ax3a.pie(oe_counts, labels=oe_counts.index, autopct='%1.1f%%', startangle=90, 
                     colors=sns.color_palette("pastel"))
            ax3a.set_title("홀:짝 비율")
            
            hl_counts = df['고저비율'].value_counts().head(5)
            ax3b.pie(hl_counts, labels=hl_counts.index, autopct='%1.1f%%', startangle=90, 
                     colors=sns.color_palette("Set2"))
            ax3b.set_title("저:고 비율")
        self._embed_graph(fig3, scroll_frame)

        # --- 그래프 4: AC값 ---
        self._add_report_section(scroll_frame, "4. 복잡도(AC값) 분석")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        if 'AC값' in df.columns:
            sns.countplot(x='AC값', data=df, ax=ax4, palette="magma")
            ax4.set_title("AC값 분포 (높을수록 무작위성 높음)")
        self._embed_graph(fig4, scroll_frame)

    def _add_report_section(self, parent, title_text):
        """리포트 소제목 추가 헬퍼 함수"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(20, 5))
        ctk.CTkLabel(frame, text=title_text, font=("Arial", 16, "bold"), 
                     text_color="#4DB6AC", anchor="w").pack(fill="x")

    def _embed_graph(self, fig, parent_widget):
        """그래프를 캔버스에 넣어 위젯에 붙이는 헬퍼 함수"""
        fig.tight_layout()
        # 그래프 배경색을 투명하게 하거나 위젯 색과 맞춤
        fig.patch.set_facecolor('#2b2b2b') 
        
        canvas = FigureCanvasTkAgg(fig, master=parent_widget)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, pady=5)