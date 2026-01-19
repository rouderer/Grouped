"""
gui.py - 图形用户界面模块
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core import CombinatorialTestingSO
from utils import calculate_coverage
import threading


class CTOSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐍 基于蛇优化算法的组合测试用例生成器")
        self.root.geometry("1000x700")
        self.setup_ui()

    def setup_ui(self):
        # ============ 样式配置 ============
        style = ttk.Style()
        style.configure("TButton", padding=6, font=("微软雅黑", 10))
        style.configure("TLabel", font=("微软雅黑", 10))

        # ============ 顶部配置区 ============
        frame_top = ttk.LabelFrame(self.root, text="1. 系统参数配置", padding=10)
        frame_top.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_top, text="参数配置 (用逗号分隔, 例如: 3,3,4,5,2):").grid(row=0, column=0, sticky="w")
        self.entry_factors = ttk.Entry(frame_top, width=50, font=("Consolas", 10))
        self.entry_factors.grid(row=0, column=1, padx=5)
        self.entry_factors.insert(0, "3, 3, 4, 5, 2")  # 默认值

        ttk.Label(frame_top, text="种群大小:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.entry_pop = ttk.Entry(frame_top, width=10)
        self.entry_pop.grid(row=1, column=1, sticky="w", padx=5, pady=(5, 0))
        self.entry_pop.insert(0, "50")

        ttk.Label(frame_top, text="最大迭代:").grid(row=1, column=2, sticky="w", pady=(5, 0), padx=(20, 0))
        self.entry_iter = ttk.Entry(frame_top, width=10)
        self.entry_iter.grid(row=1, column=3, sticky="w", padx=5, pady=(5, 0))
        self.entry_iter.insert(0, "100")

        # ============ 中部操作区 ============
        frame_action = ttk.Frame(self.root)
        frame_action.pack(fill="x", padx=10, pady=5)

        btn_run = ttk.Button(frame_action, text="🚀 开始优化", command=self.start_optimization)
        btn_run.pack(side="left", padx=10)

        self.progress = ttk.Progressbar(frame_action, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        # ============ 结果展示区 (分页) ============
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # --- 选项卡1: 用例表格 ---
        frame_table = ttk.Frame(self.notebook)
        self.notebook.add(frame_table, text="📋 生成的测试用例")

        self.tree = ttk.Treeview(frame_table, columns=[], show="headings")
        scrollbar_y = ttk.Scrollbar(frame_table, orient="vertical", command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(frame_table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")

        # --- 选项卡2: 日志输出 ---
        frame_log = ttk.Frame(self.notebook)
        self.notebook.add(frame_log, text="📝 运行日志")

        self.text_log = scrolledtext.ScrolledText(frame_log, state='disabled', height=20)
        self.text_log.pack(fill="both", expand=True, padx=5, pady=5)

        # --- 选项卡3: 收敛曲线 ---
        frame_plot = ttk.Frame(self.notebook)
        self.notebook.add(frame_plot, text="📈 收敛曲线")

        self.plot_frame = ttk.Frame(frame_plot)
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def log_message(self, message):
        """向日志窗口追加消息"""
        self.text_log.config(state='normal')
        self.text_log.insert('end', message + '\n')
        self.text_log.see('end')  # 自动滚动到底部
        self.text_log.config(state='disabled')

    def start_optimization(self):
        """启动优化线程"""
        try:
            # 1. 获取输入
            factors_str = self.entry_factors.get()
            factors = [int(x.strip()) for x in factors_str.split(",") if x.strip()]

            pop_size = int(self.entry_pop.get())
            max_iter = int(self.entry_iter.get())

            if not factors:
                raise ValueError("参数配置不能为空")

            # 2. 禁用按钮，显示进度
            self.progress.start()
            for child in self.root.winfo_children():
                if isinstance(child, ttk.Button):
                    child.state(['disabled'])

            # 3. 开启后台线程执行计算 (防止界面卡死)
            thread = threading.Thread(target=self.run_algorithm, args=(factors, pop_size, max_iter))
            thread.start()

        except Exception as e:
            messagebox.showerror("输入错误", f"配置解析失败: {str(e)}")

    def run_algorithm(self, factors, pop_size, max_iter):
        """在后台线程中运行算法"""
        try:
            self.log_message(f"[开始] 配置: {factors}, 种群={pop_size}, 迭代={max_iter}")

            # --- 调用核心算法 ---
            ctso = CombinatorialTestingSO(factors, t_way=2, pop_size=pop_size, max_iter=max_iter)
            # 修复：接收 core.py 返回的 3 个值
            test_suite, final_score, history = ctso.optimize()
            # --- 计算覆盖率 ---
            coverage_rate, covered, total = calculate_coverage(test_suite, factors)
            # --- 回到主线程更新界面 ---
            self.root.after(0, self.update_ui, test_suite, coverage_rate, final_score, ctso.best_fitness_history)

            self.log_message("[完成] 算法执行完毕")

        except Exception as e:
            self.root.after(0, messagebox.showerror, "运行错误", str(e))
        finally:
            # 恢复界面
            self.root.after(0, self.progress.stop)
            for child in self.root.winfo_children():
                if isinstance(child, ttk.Button):
                    child.state(['!disabled'])

    def update_ui(self, test_suite, coverage_rate, final_score, history):
        """更新UI界面 (由后台线程回调)"""

        # --- 1. 更新表格 ---
        # 清空旧列
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = [f"参数{i}" for i in range(len(test_suite[0]))] if test_suite else []

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, anchor="center")

        for i, case in enumerate(test_suite):
            self.tree.insert("", "end", values=case)

        # --- 2. 更新日志摘要 ---
        self.log_message(f"\n--- 优化结果摘要 ---")
        self.log_message(f"生成用例数: {len(test_suite)}")
        self.log_message(f"覆盖率: {coverage_rate:.2%}")
        self.log_message(f"最终适应度: {final_score:.4f}")

        # --- 3. 绘制收敛曲线 ---
        for widget in self.plot_frame.winfo_children():
            widget.destroy()  # 清空旧图

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        ax.plot(history, 'b-o', label='最优适应度')
        ax.set_title('收敛曲线')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('适应度值')
        ax.grid(True, alpha=0.3)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # 关闭 matplotlib 自己的窗口，只在 Tkinter 中显示
        plt.close(fig)


def launch_gui():
    """启动 GUI"""
    root = tk.Tk()
    app = CTOSApp(root)
    root.mainloop()