# analyze_fewshot_bias.py
import pandas as pd
import numpy as np
import os

# ==================== 配置 ====================
BASE = "/workspace/QWEN2.5_FEWSHOT_MO2"
MODEL = "gpt-3-noperception-reflection-1-100agents-240months"
DATA = os.path.join(BASE, "data", MODEL) 
OUT = os.path.join(DATA, "fewshot_analysis")
os.makedirs(OUT, exist_ok=True)

# ==================== 加载数据 ====================
csv_path = os.path.join(DATA, "good_decisions.csv")
print(f"📂 加载数据: {csv_path}")

if not os.path.exists(csv_path):
    print(f"❌ 文件不存在: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
print(f"✅ 加载成功: {len(df)} 条决策")
print(f"   列名: {list(df.columns)}")

# ==================== 1. 就业状态分析 ====================
print("\n" + "=" * 70)
print("📊 就业状态分析")
print("=" * 70)

employed = df[df['work_decision'] == 1.0]
unemployed = df[df['work_decision'] == 0.0]

print(f"\n总决策数: {len(df)}")
print(f"就业决策: {len(employed)} ({len(employed)/len(df)*100:.2f}%)")
print(f"失业决策: {len(unemployed)} ({len(unemployed)/len(df)*100:.2f}%)")

if len(employed)/len(df) > 0.9:
    print("\n⚠️ 警告：就业决策占比>90%，存在严重偏差！")
    print("   → 这解释了为什么Few-shot会导致失业率违约上升")

# ==================== 2. 就业状态 vs Score ====================
print("\n" + "=" * 70)
print("📊 就业状态 vs Score分布")
print("=" * 70)

print(f"\n就业agent的Score:")
print(f"  均值: {employed['score'].mean():.2f}")
print(f"  中位数: {employed['score'].median():.0f}")
print(f"  最小值: {employed['score'].min():.0f}")
print(f"  最大值: {employed['score'].max():.0f}")

if len(unemployed) > 0:
    print(f"\n失业agent的Score:")
    print(f"  均值: {unemployed['score'].mean():.2f}")
    print(f"  中位数: {unemployed['score'].median():.0f}")
    print(f"  最小值: {unemployed['score'].min():.0f}")
    print(f"  最大值: {unemployed['score'].max():.0f}")
    
    score_diff = employed['score'].mean() - unemployed['score'].mean()
    print(f"\nScore差异: {score_diff:.2f}")
    if score_diff > 2:
        print("  ⚠️ 就业agent的Score明显更高！")
        print("  → 按score筛选会严重偏向就业决策")
else:
    print(f"\n⚠️ 没有失业决策数据！所有'好决策'都是就业决策")

# ==================== 3. 高分决策的就业偏差 ====================
print("\n" + "=" * 70)
print("📊 高分决策的就业偏差分析")
print("=" * 70)

# Top 10%的决策
top10_threshold = df['score'].quantile(0.9)
top10 = df[df['score'] >= top10_threshold]

print(f"\nTop 10% 决策 (score >= {top10_threshold:.0f}):")
print(f"  总数: {len(top10)}")
print(f"  就业: {(top10['work_decision'] == 1.0).sum()} ({(top10['work_decision'] == 1.0).sum()/len(top10)*100:.2f}%)")
print(f"  失业: {(top10['work_decision'] == 0.0).sum()} ({(top10['work_decision'] == 0.0).sum()/len(top10)*100:.2f}%)")

# Top 20%的决策
top20_threshold = df['score'].quantile(0.8)
top20 = df[df['score'] >= top20_threshold]

print(f"\nTop 20% 决策 (score >= {top20_threshold:.0f}):")
print(f"  总数: {len(top20)}")
print(f"  就业: {(top20['work_decision'] == 1.0).sum()} ({(top20['work_decision'] == 1.0).sum()/len(top20)*100:.2f}%)")
print(f"  失业: {(top20['work_decision'] == 0.0).sum()} ({(top20['work_decision'] == 0.0).sum()/len(top20)*100:.2f}%)")

# Score=最高分的决策
max_score = df['score'].max()
max_score_decisions = df[df['score'] == max_score]

print(f"\n最高分决策 (score = {max_score:.0f}):")
print(f"  总数: {len(max_score_decisions)}")
print(f"  就业: {(max_score_decisions['work_decision'] == 1.0).sum()} ({(max_score_decisions['work_decision'] == 1.0).sum()/len(max_score_decisions)*100:.2f}%)")
print(f"  失业: {(max_score_decisions['work_decision'] == 0.0).sum()} ({(max_score_decisions['work_decision'] == 0.0).sum()/len(max_score_decisions)*100:.2f}%)")

# ==================== 4. Score分段分析 ====================
print("\n" + "=" * 70)
print("📊 Score分段的就业率")
print("=" * 70)

score_bins = [10, 12, 14, 16, 20]
df['score_bin'] = pd.cut(df['score'], bins=score_bins, labels=['10-12', '12-14', '14-16', '16-20'])
employment_by_score = df.groupby('score_bin', observed=True)['work_decision'].agg(['mean', 'count'])

print("\nScore范围   就业率   样本数")
print("-" * 40)
for idx, row in employment_by_score.iterrows():
    print(f"{idx:10s}  {row['mean']*100:5.1f}%   {int(row['count']):6d}")

# ==================== 5. Few-shot实际使用的示例分析 ====================
print("\n" + "=" * 70)
print("📊 Few-shot实际使用示例分析（模拟）")
print("=" * 70)

def simulate_fewshot_selection(agent_income, agent_wealth, df, n=3):
    """模拟Few-shot示例选择"""
    income_low = agent_income * 0.5
    income_high = agent_income * 2.0
    wealth_low = agent_wealth * 0.5
    wealth_high = agent_wealth * 2.0
    
    # 层级1：收入+财富筛选
    candidates = df[
        (df['curr_income'] >= income_low) &
        (df['curr_income'] <= income_high) &
        (df['curr_wealth'] >= wealth_low) &
        (df['curr_wealth'] <= wealth_high)
    ].copy()
    
    # 层级2：仅收入筛选
    if len(candidates) < n:
        candidates = df[
            (df['curr_income'] >= income_low) &
            (df['curr_income'] <= income_high)
        ].copy()
    
    # 层级3：全局
    if len(candidates) < n:
        candidates = df.copy()
    
    # 按score取top-n
    if len(candidates) > 0:
        return candidates.nlargest(min(n, len(candidates)), 'score')
    else:
        return candidates

# 测试不同收入水平的agent
test_cases = [
    {"name": "低收入", "income": 3000, "wealth": 5000},
    {"name": "中等收入", "income": 15000, "wealth": 20000},
    {"name": "高收入", "income": 50000, "wealth": 80000},
]

print("\nAgent类型      匹配示例数  就业示例  就业率   Score范围")
print("-" * 70)

for case in test_cases:
    examples = simulate_fewshot_selection(case['income'], case['wealth'], df)
    if len(examples) > 0:
        employed_count = (examples['work_decision'] == 1.0).sum()
        score_min = examples['score'].min()
        score_max = examples['score'].max()
        
        print(f"{case['name']:10s}  {len(examples):6d}      {employed_count:6d}    {employed_count/len(examples)*100:5.1f}%   {score_min:.0f}-{score_max:.0f}")

# ==================== 6. 消费倾向分析 ====================
print("\n" + "=" * 70)
print("📊 消费倾向分析")
print("=" * 70)

print(f"\n整体消费倾向:")
print(f"  均值: {df['consumption_prop'].mean():.3f}")
print(f"  中位数: {df['consumption_prop'].median():.3f}")
print(f"  最小值: {df['consumption_prop'].min():.3f}")
print(f"  最大值: {df['consumption_prop'].max():.3f}")

print(f"\n就业vs失业的消费倾向:")
print(f"  就业agent平均消费倾向: {employed['consumption_prop'].mean():.3f}")
if len(unemployed) > 0:
    print(f"  失业agent平均消费倾向: {unemployed['consumption_prop'].mean():.3f}")

# ==================== 7. 关键发现总结 ====================
print("\n" + "=" * 70)
print("🔍 关键发现总结")
print("=" * 70)

employment_ratio = len(employed) / len(df)
top10_employment_ratio = (top10['work_decision'] == 1.0).sum() / len(top10)

print(f"\n1. 整体就业偏差:")
print(f"   - 所有决策中就业率: {employment_ratio*100:.1f}%")
if employment_ratio > 0.9:
    print(f"   - ⚠️ 严重偏向就业！")

print(f"\n2. 高分决策偏差:")
print(f"   - Top 10%决策中就业率: {top10_employment_ratio*100:.1f}%")
if top10_employment_ratio > 0.95:
    print(f"   - ⚠️ 高分决策几乎全是就业！")

print(f"\n3. Few-shot影响:")
print(f"   - Few-shot按score选择示例")
print(f"   - 导致agent看到的都是就业示例")
print(f"   - 结果：agent不知道什么时候该失业")
print(f"   - 这就是失业率违约从38.75%升到69.58%的原因")

print(f"\n4. 改进建议:")
print(f"   a) 不只按score筛选，也要平衡就业/失业比例")
print(f"   b) 添加'合理失业'的示例（收入太低时拒绝工作）")
print(f"   c) 按约束维度分别筛选示例")

# ==================== 8. 保存结果 ====================
summary_path = os.path.join(OUT, "fewshot_bias_summary.csv")
with open(summary_path, 'w') as f:
    f.write("metric,value\n")
    f.write(f"total_decisions,{len(df)}\n")
    f.write(f"employed_decisions,{len(employed)}\n")
    f.write(f"unemployed_decisions,{len(unemployed)}\n")
    f.write(f"employed_ratio,{employment_ratio}\n")
    f.write(f"employed_avg_score,{employed['score'].mean()}\n")
    if len(unemployed) > 0:
        f.write(f"unemployed_avg_score,{unemployed['score'].mean()}\n")
    f.write(f"top10_employed_ratio,{top10_employment_ratio}\n")
    f.write(f"top20_employed_ratio,{(top20['work_decision'] == 1.0).sum()/len(top20)}\n")
    f.write(f"max_score,{max_score}\n")

print(f"\n✅ 统计结果已保存: {summary_path}")

print("\n" + "=" * 70)
print("✅ 分析完成！")
print("=" * 70)