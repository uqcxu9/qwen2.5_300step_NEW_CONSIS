import pickle as pkl
import numpy as np

path = r"C:\Users\徐瑞岑\OneDrive\PHD\experiment\QWEN2.5\QWEN2.5_BUFFER_RESULT-main\data\gpt-3-noperception-reflection-1-100agents-240months\dense_log.pkl"

with open(path, "rb") as f:
    log = pkl.load(f)

states = log["states"]

T = len(states)
num_agents = len([k for k in states[0].keys() if k != 'p'])

# 计算每月平均 skill（工资率）
monthly_avg_skill = []
for t in range(T):
    skills = [states[t][str(i)]['skill'] for i in range(num_agents)]
    monthly_avg_skill.append(np.mean(skills))

# 年度 wage rate inflation（用于 Phillips Curve）
print("=== 年度 Wage Rate Inflation ===")
annual_wage_inflation = []
for year in range(1, T // 12):
    avg_this = np.mean(monthly_avg_skill[year*12 : (year+1)*12])
    avg_last = np.mean(monthly_avg_skill[(year-1)*12 : year*12])
    wage_inflation = (avg_this - avg_last) / avg_last
    annual_wage_inflation.append(wage_inflation)
    print(f"Year {year+1}: {wage_inflation*100:.2f}%")

# 如果也想看月度变化
print("\n=== 月度 Wage Rate Inflation (前24个月) ===")
for t in range(1, min(25, T)):
    m = (monthly_avg_skill[t] - monthly_avg_skill[t-1]) / (monthly_avg_skill[t-1] + 1e-8)
    print(f"Month {t}: {m*100:.2f}%")