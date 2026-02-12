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

# 로직 모듈
from src.data_loader import DataLoader
from src.predict_lotto import LottoAI

# ==========================================
# 폰트 설정 (한글 깨짐 방지)
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


class LottoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Integrated Lotto Predictor")
        self.geometry("850x900")
        
        self.loader = DataLoader()
        self.ai = LottoAI()
        
        # X 버튼 클릭 시 종료 이벤트 연결
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self._init_ui()

    def _init_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # === 1. 헤더 영역 ===
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.header_frame, text="AI 복권 분석 & 예측 시스템", 
                     font=("Arial", 24, "bold")).pack(pady=10)
        
        # [모드 선택]
        self.mode_var = ctk.StringVar(value="로또 6/45")
        self.combo_mode = ctk.CTkOptionMenu(
            self.header_frame, 
            values=["로또 6/45", "연금복권 720+"],
            variable=self.mode_var,
            command=self.change_mode_ui
        )
        self.combo_mode.pack(padx=20, pady=(0, 5), fill="x")

        # [파일 로드 버튼]
        self.btn_file = ctk.CTkButton(self.header_frame, text="📂 데이터 파일 열기 (Excel/CSV)", command=self.load_file)
        self.btn_file.pack(padx=20, pady=(0, 10), fill="x")

        self.lbl_status = ctk.CTkLabel(self.header_frame, text="파일 없음", text_color="gray")
        self.lbl_status.pack(pady=(0, 5))

        # === 2. 설정 영역 ===
        self.setting_frame = ctk.CTkFrame(self)
        self.setting_frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        self.setting_frame.grid_columnconfigure((0, 1), weight=1)

        # 게임 수
        ctk.CTkLabel(self.setting_frame, text="생성할 게임 수:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_count = ctk.CTkEntry(self.setting_frame, placeholder_text="예: 5")
        self.entry_count.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.entry_count.insert(0, "5")

        # 고정수
        ctk.CTkLabel(self.setting_frame, text="고정수 (로또 전용):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_fixed = ctk.CTkEntry(self.setting_frame, placeholder_text="예: 7, 15")
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
        self.log_textbox.insert("0.0", "시스템 준비 완료.\n")
        self.log_textbox.configure(state="disabled")

    def log(self, msg):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", msg + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def on_closing(self):
        """프로그램 완전 종료"""
        self.destroy()
        os._exit(0)

    def change_mode_ui(self, choice):
        self.log(f"🔄 모드 변경됨: {choice}")
        self.lbl_status.configure(text="파일을 다시 로드해주세요.", text_color="gray")
        # 모드 변경 시 기존 데이터와의 충돌 방지를 위해 버튼 비활성화
        self.btn_analyze.configure(state="disabled")
        self.btn_predict.configure(state="disabled")
        self.loader.df = None # 데이터 초기화

    def load_file(self):
        """파일을 로드하고 현재 모드에 맞는지 즉시 검사합니다."""
        path = filedialog.askopenfilename(filetypes=[("Data Files", "*.xlsx *.csv")])
        if not path:
            return

        try:
            # 1. 현재 모드 확인
            mode_str = self.mode_var.get()
            mode_code = "lotto" if mode_str == "로또 6/45" else "pension"
            
            # 2. 파일 로드 시도
            self.loader.load_file(path, mode=mode_code)
            
            # 3. [핵심] 컬럼 검사 (Validation)
            df = self.loader.df
            if df is None:
                raise Exception("파일을 읽을 수 없습니다.")

            if mode_code == "lotto":
                # 로또 필수 컬럼 확인
                required = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
                if not all(col in df.columns for col in required):
                    raise ValueError(f"선택한 모드는 [{mode_str}]인데,\n파일 형식이 맞지 않습니다.\n(연금복권 파일인가요?)")
            else:
                # 연금복권 필수 컬럼 확인
                required = ['조', '번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
                if not all(col in df.columns for col in required):
                    raise ValueError(f"선택한 모드는 [{mode_str}]인데,\n파일 형식이 맞지 않습니다.\n(로또 파일인가요?)")

            # 4. 검사 통과 시 버튼 활성화
            self.lbl_status.configure(text=f"로드 완료: {os.path.basename(path)}", text_color="#66BB6A")
            self.btn_analyze.configure(state="normal")
            self.btn_predict.configure(state="normal")
            self.log(f"[시스템] {mode_str} 데이터 로드 성공! ({len(df)}행)")

        except ValueError as ve:
            # 데이터 형식이 안 맞을 때 (경고창 + 버튼 비활성화)
            messagebox.showerror("데이터 불일치", str(ve))
            self.lbl_status.configure(text="파일 형식 불일치", text_color="#FF5252")
            self.btn_analyze.configure(state="disabled")
            self.btn_predict.configure(state="disabled")
            self.loader.df = None # 잘못된 데이터 비우기
            self.log(f"[경고] {ve}")
            
        except Exception as e:
            # 기타 에러
            self.log(f"[에러] {e}")
            messagebox.showerror("오류", f"파일 로드 중 오류가 발생했습니다.\n{e}")

    def start_thread(self):
        self.btn_predict.configure(state="disabled", text="학습 중...")
        threading.Thread(target=self.run_ai).start()

    def run_ai(self):
        try:
            mode_str = self.mode_var.get()
            mode_code = "lotto" if mode_str == "로또 6/45" else "pension"
            
            # (이중 안전장치) 데이터 확인
            if self.loader.df is None:
                raise Exception("데이터가 로드되지 않았습니다.")

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
            if data is None: raise Exception("데이터 전처리 실패")
            
            self.ai.train_model(data, mode=mode_code, epochs=100)
            self.log(">>> 모델 학습 완료! 번호 생성 중...")

            last_data = data[-self.ai.window_size:]
            results = []
            if mode_code == "lotto":
                past_combos = self.loader.get_past_combinations()
                results = self.ai.predict_lotto(last_data, past_combos, count=game_count, fixed_numbers=fixed_nums)
            else:
                results = self.ai.predict_pension(last_data, count=game_count)

            self.log(f"\n====== {mode_str} AI 추천 ======")
            for i, res in enumerate(results):
                if mode_code == "pension":
                    self.log(f" GAME {i+1}:  [{res[0]}조]  {' '.join(map(str, res[1:]))}")
                else:
                    self.log(f" GAME {i+1}:  {res}  (합: {sum(res)})")
            self.log("================================")

        except Exception as e:
            self.log(f"[오류] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_predict.configure(state="normal", text="🔮 AI 예측 시작")

    def show_analysis(self):
        # 버튼이 활성화되어 있다면 이미 load_file에서 검증된 상태임
        mode_str = self.mode_var.get()
        if mode_str == "연금복권 720+":
             self.show_pension_analysis()
        else:
             self.show_lotto_analysis()

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
        if '조' in df.columns:
            sns.countplot(x='조', data=df, ax=ax1, palette="viridis")
        self._embed_graph(fig1, scroll_frame)
        
        self._add_report_section(scroll_frame, "2. 각 자리별 숫자(0~9) 출현 빈도 Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        heatmap_data = np.zeros((6, 10))
        cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            for i, col in enumerate(valid_cols):
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