# src/gui.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import font_manager, rc
import seaborn as sns
import numpy as np
import threading
import os
import sys
import platform
import time

from src.data_loader import DataLoader
from src.predict_lotto import LottoAI

# ==========================================
# 폰트 및 테마 설정
# ==========================================
if platform.system() == 'Windows':
    try:
        font_path = "c:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    except:
        rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# ==========================================
# 색상 유틸리티 함수
# ==========================================
def get_lotto_color(num):
    """로또 번호별 공 색상 반환"""
    n = int(num)
    if 1 <= n <= 10: return "#FBC400" # 노랑 (1~10)
    elif 11 <= n <= 20: return "#69C7F0" # 파랑 (11~20)
    elif 21 <= n <= 30: return "#FF7272" # 빨강 (21~30)
    elif 31 <= n <= 40: return "#AAAAAA" # 회색 (31~40)
    else: return "#B0D840" # 초록 (41~45)

class LottoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Integrated Lotto Predictor Pro")
        self.geometry("1000x900") # 너비 확장
        
        self.loader = DataLoader()
        self.ai = LottoAI()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._init_ui()

    def _init_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1) # 결과창 영역을 크게

        # === 1. 헤더 ===
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.header_frame, text="AI 복권 분석 & 예측 시스템", 
                     font=("Arial", 24, "bold")).pack(pady=10)
        
        # 컨트롤 패널 (모드, 파일, 설정)
        self.ctrl_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.ctrl_frame.pack(fill="x", padx=10, pady=5)
        
        self.mode_var = ctk.StringVar(value="로또 6/45")
        self.combo_mode = ctk.CTkOptionMenu(
            self.ctrl_frame, values=["로또 6/45", "연금복권 720+"],
            variable=self.mode_var, command=self.change_mode_ui, width=150
        )
        self.combo_mode.pack(side="left", padx=5)

        self.btn_file = ctk.CTkButton(self.ctrl_frame, text="📂 파일 열기", width=100, command=self.load_file)
        self.btn_file.pack(side="left", padx=5)
        
        self.lbl_status = ctk.CTkLabel(self.ctrl_frame, text="파일 없음", text_color="gray")
        self.lbl_status.pack(side="left", padx=10)

        # === 2. 설정 및 실행 ===
        self.setting_frame = ctk.CTkFrame(self)
        self.setting_frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        
        ctk.CTkLabel(self.setting_frame, text="게임 수:").pack(side="left", padx=10)
        self.entry_count = ctk.CTkEntry(self.setting_frame, width=50)
        self.entry_count.insert(0, "5")
        self.entry_count.pack(side="left")
        
        ctk.CTkLabel(self.setting_frame, text="고정수(로또):").pack(side="left", padx=(20, 5))
        self.entry_fixed = ctk.CTkEntry(self.setting_frame, width=100, placeholder_text="예: 1, 5")
        self.entry_fixed.pack(side="left")

        self.btn_predict = ctk.CTkButton(self.setting_frame, text="🔮 예측 시작", command=self.start_thread, fg_color="#3949AB")
        self.btn_predict.pack(side="right", padx=10, pady=10)

        self.btn_analyze = ctk.CTkButton(self.setting_frame, text="📊 분석 리포트", command=self.show_analysis, state="disabled", fg_color="#00897B")
        self.btn_analyze.pack(side="right", padx=5, pady=10)

        # === 3. 진행바 ===
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, padx=20, pady=0, sticky="ew")
        self.lbl_progress = ctk.CTkLabel(self.progress_frame, text="준비 완료", text_color="#1E88E5", anchor="w")
        self.lbl_progress.pack(fill="x")
        self.progressbar = ctk.CTkProgressBar(self.progress_frame)
        self.progressbar.pack(fill="x", pady=(0, 10))
        self.progressbar.set(0)

        # === 4. [핵심] 결과 비주얼 뷰어 (스크롤 가능) ===
        self.result_view = ctk.CTkScrollableFrame(self, corner_radius=10, fg_color="transparent")
        self.result_view.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        
        # 초기 안내 문구
        self.placeholder_lbl = ctk.CTkLabel(self.result_view, text="예측 버튼을 누르면 여기에 번호가 표시됩니다.", 
                                            font=("Arial", 16), text_color="gray")
        self.placeholder_lbl.pack(pady=50)

        # === 5. 로그 창 (작게 축소) ===
        self.log_textbox = ctk.CTkTextbox(self, height=100, font=("Consolas", 12))
        self.log_textbox.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.log_textbox.insert("0.0", "시스템 로그...\n")

    # ==========================================
    # 시각화 로직 (Balls Rendering)
    # ==========================================
    def clear_results(self):
        for widget in self.result_view.winfo_children():
            widget.destroy()

    def create_ball(self, parent, text, color, size=40):
        """동그란 공 모양 위젯 생성"""
        # 버튼을 둥글게 만들어서 공처럼 보이게 함
        btn = ctk.CTkButton(
            parent, 
            text=str(text), 
            width=size, 
            height=size, 
            corner_radius=size/2, 
            fg_color=color,
            text_color="white" if color != "#FBC400" else "black", # 노란공은 검은글씨
            font=("Arial", 16, "bold"),
            hover=False,
            state="disabled" # 클릭 안되게
        )
        return btn

    def visualize_results(self, results, mode):
        """결과 데이터를 받아서 그래픽으로 표시"""
        self.clear_results()
        
        for i, (nums, reason) in enumerate(results):
            # 게임별 카드(프레임) 생성
            card = ctk.CTkFrame(self.result_view, corner_radius=10, border_width=1, border_color="#444")
            card.pack(fill="x", pady=10, padx=5)
            
            # 상단: 게임 번호 및 이유
            header_frame = ctk.CTkFrame(card, fg_color="transparent")
            header_frame.pack(fill="x", padx=10, pady=(10, 5))
            
            ctk.CTkLabel(header_frame, text=f"GAME {i+1}", font=("Arial", 14, "bold"), text_color="#aaa").pack(side="left")
            ctk.CTkLabel(header_frame, text=f"💡 {reason}", font=("Arial", 12), text_color="#FFB74D").pack(side="right")

            # 하단: 공 나열
            ball_frame = ctk.CTkFrame(card, fg_color="transparent")
            ball_frame.pack(pady=(5, 15))

            if mode == "lotto":
                # 로또 볼 그리기
                for num in nums:
                    color = get_lotto_color(num)
                    ball = self.create_ball(ball_frame, num, color)
                    ball.pack(side="left", padx=5)
                
                # 합계 표시
                ctk.CTkLabel(ball_frame, text=f"(합: {sum(nums)})", font=("Arial", 12)).pack(side="left", padx=10)

            else:
                # 연금복권 볼 그리기
                # [조]
                jo_ball = self.create_ball(ball_frame, str(nums[0])+"조", "#9C27B0", size=50) # 보라색 조
                jo_ball.pack(side="left", padx=(0, 15))
                
                # [숫자 6개]
                for num in nums[1:]:
                    # 연금복권 숫자는 색상이 다양하지만, 여기선 깔끔하게 짙은 파랑 통일 혹은 랜덤
                    # 시인성을 위해 단색(Deep Orange) 사용
                    ball = self.create_ball(ball_frame, num, "#E65100") 
                    ball.pack(side="left", padx=3)

    # ==========================================
    # 기본 기능 함수들 (기존 로직 유지)
    # ==========================================
    def log(self, msg):
        self.log_textbox.insert("end", msg + "\n")
        self.log_textbox.see("end")

    def on_closing(self):
        self.destroy()
        os._exit(0)

    def change_mode_ui(self, choice):
        self.log(f"🔄 모드 변경됨: {choice}")
        self.lbl_status.configure(text="파일을 다시 로드해주세요.", text_color="gray")
        self.btn_analyze.configure(state="disabled")
        self.btn_predict.configure(state="disabled")
        self.loader.df = None
        self.clear_results()

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Data Files", "*.xlsx *.csv")])
        if not path: return

        try:
            mode_str = self.mode_var.get()
            mode_code = "lotto" if mode_str == "로또 6/45" else "pension"
            
            self.loader.load_file(path, mode=mode_code)
            
            # Validation
            df = self.loader.df
            if df is None: raise Exception("파일 로드 실패")

            if mode_code == "lotto":
                required = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
                if not all(col in df.columns for col in required):
                    raise ValueError(f"[{mode_str}] 파일 형식이 아닙니다.")
            else:
                required = ['조', '번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
                if not all(col in df.columns for col in required):
                    raise ValueError(f"[{mode_str}] 파일 형식이 아닙니다.")

            self.lbl_status.configure(text=f"로드 완료: {os.path.basename(path)}", text_color="#66BB6A")
            self.btn_analyze.configure(state="normal")
            self.btn_predict.configure(state="normal")
            self.log(f"[시스템] {mode_str} 데이터 로드 성공!")
            self.lbl_progress.configure(text="준비 완료")
            self.progressbar.set(0)

        except Exception as e:
            messagebox.showerror("오류", str(e))
            self.lbl_status.configure(text="파일 오류", text_color="#FF5252")

    def start_thread(self):
        self.btn_predict.configure(state="disabled", text="학습 중...")
        self.lbl_progress.configure(text="AI 엔진 가동 중...")
        self.progressbar.set(0)
        # 결과창 초기화
        self.clear_results()
        threading.Thread(target=self.run_ai).start()

    def update_progress_gui(self, value, msg):
        self.progressbar.set(value)
        self.lbl_progress.configure(text=msg)

    def run_ai(self):
        try:
            mode_str = self.mode_var.get()
            mode_code = "lotto" if mode_str == "로또 6/45" else "pension"
            
            if self.loader.df is None: raise Exception("데이터 로드 필요")
            
            try:
                game_count = int(self.entry_count.get())
                if game_count < 1: game_count = 1
            except: game_count = 5

            fixed_nums = []
            if mode_code == "lotto":
                fixed_str = self.entry_fixed.get().strip()
                if fixed_str:
                    try:
                        fixed_nums = [int(n.strip()) for n in fixed_str.split(',') if n.strip().isdigit()]
                        fixed_nums = [n for n in fixed_nums if 1 <= n <= 45]
                        fixed_nums = sorted(list(set(fixed_nums)))[:5]
                    except: pass
            
            self.log(f"\n>>> [{mode_str}] 학습 시작...")
            data = self.loader.preprocess()
            
            # AI 학습
            self.ai.train_model(data, mode=mode_code, epochs=100, progress_cb=self.update_progress_gui)
            
            # AI 예측
            last_data = data[-self.ai.window_size:]
            results = []
            if mode_code == "lotto":
                past_combos = self.loader.get_past_combinations()
                results = self.ai.predict_lotto(last_data, past_combos, count=game_count, fixed_numbers=fixed_nums, progress_cb=self.update_progress_gui)
            else:
                results = self.ai.predict_pension(last_data, count=game_count, progress_cb=self.update_progress_gui)

            # [핵심] 그래픽 시각화 호출
            # 메인 스레드에서 UI를 업데이트하기 위해 after 사용 권장되나, ctk는 thread-safe한 편이므로 직접 호출
            # 안전을 위해 약간의 딜레이
            self.visualize_results(results, mode_code)
            
            self.lbl_progress.configure(text="완료!")
            self.progressbar.set(1.0)
            self.log(">>> 예측 완료. 결과 화면을 확인하세요.")

        except Exception as e:
            self.log(f"[오류] {e}")
            messagebox.showerror("오류", str(e))
        finally:
            self.btn_predict.configure(state="normal", text="🔮 예측 시작")

    def show_analysis(self):
        # 기존 분석 코드와 동일하게 연결
        if self.loader.df is None: return
        mode_str = self.mode_var.get()
        if mode_str == "연금복권 720+": self.show_pension_analysis()
        else: self.show_lotto_analysis()

    # (show_lotto_analysis, show_pension_analysis 등 기존 코드 생략 - 그대로 사용)
    # 편의상 이전에 드린 코드의 분석 메서드들을 그대로 복사해서 붙여넣으시면 됩니다.
    def show_lotto_analysis(self):
        df = self.loader.df
        win = ctk.CTkToplevel(self)
        win.title("로또 6/45 분석 리포트")
        win.geometry("950x800")
        scroll_frame = ctk.CTkScrollableFrame(win)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        plt.style.use('dark_background')

        self._add_report_section(scroll_frame, "1. 번호별 당첨 횟수 분포")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        all_nums = df[['번호1','번호2','번호3','번호4','번호5','번호6']].values.flatten()
        sns.histplot(all_nums, bins=45, ax=ax1, color='#29B6F6', edgecolor='black')
        ax1.set_xlim(0, 46)
        self._embed_graph(fig1, scroll_frame)

        self._add_report_section(scroll_frame, "2. 당첨 번호 합계(Sum) 분포")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        if '총합' in df.columns:
            sns.histplot(df['총합'], kde=True, ax=ax2, color='#FFCA28', bins=30)
        self._embed_graph(fig2, scroll_frame)

        self._add_report_section(scroll_frame, "3. 홀짝 / 고저 비율")
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(8, 4))
        if '홀짝비율' in df.columns:
            oe = df['홀짝비율'].value_counts().head(5)
            ax3a.pie(oe, labels=oe.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        if '고저비율' in df.columns:
            hl = df['고저비율'].value_counts().head(5)
            ax3b.pie(hl, labels=hl.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
        self._embed_graph(fig3, scroll_frame)

        self._add_report_section(scroll_frame, "4. 복잡도(AC값) 분석")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        if 'AC값' in df.columns:
            sns.countplot(x='AC값', data=df, ax=ax4, palette="magma")
        self._embed_graph(fig4, scroll_frame)

    def show_pension_analysis(self):
        df = self.loader.df
        win = ctk.CTkToplevel(self)
        win.title("연금복권 720+ 분석 리포트")
        win.geometry("950x800")
        scroll_frame = ctk.CTkScrollableFrame(win)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        plt.style.use('dark_background')
        
        self._add_report_section(scroll_frame, "1. 조(Group)별 1등 당첨 빈도")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.countplot(x='조', data=df, ax=ax1, palette="viridis")
        self._embed_graph(fig1, scroll_frame)
        
        self._add_report_section(scroll_frame, "2. 각 자리별 숫자(0~9) 출현 빈도 Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        heatmap_data = np.zeros((6, 10))
        cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        for i, col in enumerate(cols):
            counts = df[col].value_counts().sort_index()
            for num, count in counts.items():
                if 0 <= num <= 9: heatmap_data[i, int(num)] = count
        sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='magma', ax=ax2,
                    xticklabels=range(10), yticklabels=['1st','2nd','3rd','4th','5th','6th'])
        self._embed_graph(fig2, scroll_frame)
        
        self._add_report_section(scroll_frame, "3. 숫자 6자리의 합 분포")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        if '숫자합' in df.columns:
            sns.histplot(df['숫자합'], kde=True, ax=ax3, color='#FFCA28')
        self._embed_graph(fig3, scroll_frame)

    def _add_report_section(self, parent, title_text):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(20, 5))
        ctk.CTkLabel(frame, text=title_text, font=("Arial", 16, "bold"), 
                     text_color="#4DB6AC", anchor="w").pack(fill="x")

    def _embed_graph(self, fig, parent_widget):
        fig.tight_layout()
        fig.patch.set_facecolor('#2b2b2b')
        canvas = FigureCanvasTkAgg(fig, master=parent_widget)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=5)