import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
import optuna
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import deepchem as dc



# ==============================================
# 1️⃣ 计算描述符
# ==============================================
def calc_descriptors_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"❌ 无法解析 SMILES: {smiles}")

    desc_values = []

    # -------- RDKit 部分（2个）--------
    desc_values.append(Descriptors.MaxAbsEStateIndex(mol))
    desc_values.append(Descriptors.SlogP_VSA1(mol))
    desc_values.append(Descriptors.SlogP_VSA11(mol))

    # -------- Mordred 部分（17个）--------
    mordred_names = [
        "ATSC7d", "AETA_beta_s", "SIC3", "SIC4", "CIC3", "ZMIC4", "AMID_O", "GGI5", "GGI8", "JGI6", "JGI8", "JGI9"
    ]
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas([mol])

    for name in mordred_names:
        if name not in df.columns:
            raise KeyError(f"❌ Mordred 未计算出描述符: {name}")
        desc_values.append(df.loc[0, name])

    return np.array(desc_values, dtype=float)


# ==============================================
# 2️⃣ 主GUI逻辑
# ==============================================
class StrengthGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Elongation Predictor")
        self.root.geometry("400x220")
        self.smiles = ""
        self.tg = None
        self.n_trials = 0
        self.model = joblib.load("./Elongation_model.pkl")  # 强度预测模型
        self.create_smiles_window()

    # ========== 第一个窗口（输入SMILES） ==========
    def create_smiles_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="请输入SMILES：", font=("Microsoft YaHei", 12)).pack(pady=10)
        self.smiles_entry = tk.Entry(self.root, width=40)
        self.smiles_entry.pack()
        tk.Button(self.root, text="确定", command=self.process_smiles).pack(pady=10)

    def process_smiles(self):
        self.smiles = self.smiles_entry.get().strip()
        if not self.smiles:
            messagebox.showerror("错误", "SMILES不能为空！")
            return

        try:
            # 计算21个描述符（不含tg）
            desc_values = calc_descriptors_from_smiles(self.smiles)

            # 预测 Tg（通过 DeepChem 模型）
            temp_dir = os.path.join(os.getcwd(), "Temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "smiles_temp.csv")
            pd.DataFrame([self.smiles], columns=["Smiles"]).to_csv(temp_path, index=False)

            featurizer = dc.feat.ConvMolFeaturizer()
            loader = dc.data.CSVLoader(tasks=[], feature_field="Smiles", featurizer=featurizer)
            testset = loader.create_dataset(temp_path)

            tg_model_dir = './models/ToObtainPredictedValuesofPolyimides'
            tg_model = dc.models.GraphConvModel(1, mode="regression", model_dir=tg_model_dir)
            tg_model.restore()
            pred_value = tg_model.predict(testset)[0][0]
            self.tg = float(pred_value)

            # 将 tg 添加到描述符中
            self.desc21 = np.concatenate([desc_values, [self.tg]])

            # 进入下一个窗口
            self.create_trials_window()

        except Exception as e:
            messagebox.showerror("错误", f"Tg预测或描述符计算失败：{e}")

    # ========== 第二个窗口（输入运行次数） ==========
    def create_trials_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        tk.Label(self.root, text="请输入运行次数：", font=("Microsoft YaHei", 12)).pack(pady=10)
        self.trials_entry = tk.Entry(self.root, width=20)
        self.trials_entry.pack()
        tk.Button(self.root, text="确定", command=self.run_optimization).pack(pady=10)

    # ========== 运行优化部分 ==========
    def run_optimization(self):
        try:
            self.n_trials = int(self.trials_entry.get().strip())
        except ValueError:
            messagebox.showerror("错误", "请输入有效整数！")
            return

        fixed_input = self.desc21.copy()

        def objective(trial):
            x17 = trial.suggest_int("x17", 3000, 30000)
            x18 = trial.suggest_int("x18", 30, 2500)
            x19 = trial.suggest_int("x19", 25, 500)
            x20 = trial.suggest_int("x20", 5, 1500)
            x21 = trial.suggest_int("x21", 1, 10)

            # --- 约束1 ---
            if x21 == 1:
                if x20 != x18:
                    return 0
            elif x21 > 1:
                if x20 > x18:
                    return 0

            # --- 约束2 ---
            if x21 == 1:
                if x17 != x18 * x19:
                    return 0
            elif x21 > 1:
                if x17 > x18 * x19:
                    return 0
            if x21 > 1:
                if x17 < x19 * x20:
                    return 0

            variable = np.array([x17, x18, x19, x20, x21])
            x_full = np.concatenate([fixed_input, variable]).reshape(1, -1)
            y_pred = self.model.predict(x_full)
            return float(y_pred[0])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_y = study.best_value
        best_params = study.best_params
        self.show_result_window(best_y, best_params)

    # ========== 最后结果窗口 ==========
    def show_result_window(self, best_y, best_params):
        for widget in self.root.winfo_children():
            widget.destroy()
        tk.Label(self.root, text=f"预测断裂伸长率(%)：{best_y:.6f}", font=("Microsoft YaHei", 12)).pack(pady=10)

        name_map = {
            "x17": "integration",
            "x18": "time_all",
            "x19": "max_temp",
            "x20": "max_temp_time",
            "x21": "temp_num"
        }
        tk.Label(self.root, text="对应工艺信息：", font=("Microsoft YaHei", 11)).pack(pady=5)
        for k, v in best_params.items():
            tk.Label(self.root, text=f"{name_map.get(k, k)} = {v}", font=("Microsoft YaHei", 10)).pack()

        tk.Button(self.root, text="关闭", command=self.root.destroy).pack(pady=15)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = StrengthGUI()
    app.run()

