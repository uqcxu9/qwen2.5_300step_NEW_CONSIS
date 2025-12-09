# -*- coding: utf-8 -*-
"""
同时用 actions 和 states 计算失业率，对比差异
"""

import os
import pickle as pkl
import csv
import matplotlib.pyplot as plt

BASE = r"/workspace/ACL24-EconAgent/data"
MODEL = "gpt-3-noperception-reflection-1-100agents-240months"  
DATA = os.path.join(BASE, MODEL)
OUT  = os.path.join(DATA, "result_analysis")
os.makedirs(OUT, exist_ok=True)

P_DENSE = os.path.join(DATA, "dense_log.pkl")
N = 100

def get_labor(v):
    """从动作条目里取劳动决策(1/0)"""
    if isinstance(v, dict):
        return int(v.get("SimpleLabor", 0))
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        return int(v[0])
    try: 
        return int(v)
    except: 
        return 0

def compute_unemployment_by_actions(actions):
    """方法1: 基于 actions 计算失业率（当月 l=0 的人数）"""
    M = len(actions)
    monthly_unemp_rate = []
    monthly_unemp_cnt = []
    
    for m in range(M):
        month_actions = actions[m] or {}
        unemployed = 0
        total = 0
        
        for k, v in month_actions.items():
            # 排除 planner
            if k == 'p':
                continue
            
            total += 1
            labor = get_labor(v)
            
            if labor == 0:
                unemployed += 1
        
        rate = unemployed / total if total > 0 else 0
        monthly_unemp_cnt.append(unemployed)
        monthly_unemp_rate.append(rate)
        
        # 打印前5个月的详细信息
        if m < 5:
            print(f"  月{m+1}: 不工作={unemployed}/{total} = {rate*100:.2f}%")
    
    return monthly_unemp_cnt, monthly_unemp_rate

def compute_unemployment_by_states(states):
    """方法2: 基于 states 计算失业率（job='Unemployment' 的人数）
    
    失业率 = 失业人数 / 劳动力人口
    劳动力人口 = 就业人数 + 失业人数
    """
    M = len(states)
    monthly_unemp_rate = []
    monthly_unemp_cnt = []
    
    for m in range(M):
        month_states = states[m] or {}
        unemployed = 0
        employed = 0  # ✅ 新增：统计就业人数
        
        for agent_id, agent_state in month_states.items():
            if agent_id == 'p':
                continue
            
            if isinstance(agent_state, dict):
                endogenous = agent_state.get('endogenous', {})
                job = endogenous.get('job', None)
                
                if job == 'Unemployment':
                    unemployed += 1
                else:
                    employed += 1  # ✅ 有工作就算就业
        
        labor_force = employed + unemployed  # ✅ 劳动力 = 就业 + 失业
        rate = unemployed / labor_force if labor_force > 0 else 0
        monthly_unemp_cnt.append(unemployed)
        monthly_unemp_rate.append(rate)
        
        if m < 5:
            print(f"  月{m+1}: 失业={unemployed}, 就业={employed}, 劳动力={labor_force}, 失业率={rate*100:.2f}%")
    
    return monthly_unemp_cnt, monthly_unemp_rate

def compute_correct_unemployment():
    """正确的失业率计算：从第2个月开始统计（跳过初始化的第1个月）"""
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    states = dense.get("states", [])
    
    if not states:
        print("错误：没有 states 数据")
        return None, None, None
    
    monthly_unemp_rate = []
    
    for m in range(1, len(states)):
        month_states = states[m] or {}
        unemployed = 0
        employed = 0  # ✅ 新增
        
        for agent_id, agent_state in month_states.items():
            if agent_id == 'p':
                continue
            
            if isinstance(agent_state, dict):
                job = agent_state.get('endogenous', {}).get('job', None)
                if job == 'Unemployment':
                    unemployed += 1
                else:
                    employed += 1  # ✅ 有工作算就业
        
        labor_force = employed + unemployed  # ✅ 劳动力
        rate = unemployed / labor_force if labor_force > 0 else 0
        monthly_unemp_rate.append(rate)
    
    # 年度平均（注意：现在只有239个月的数据）
    years = []
    y_rates = []
    
    # 第1年：从月1到月12（索引0-11）
    for y in range(0, len(monthly_unemp_rate), 12):
        chunk = monthly_unemp_rate[y:y+12]
        if len(chunk) == 12:  # 只统计完整年份
            years.append(y//12 + 1)
            y_rates.append(sum(chunk) / 12)
    
    print("正确的年度失业率(%):", [round(x*100, 2) for x in y_rates])
    
    return monthly_unemp_rate, years, y_rates

def compute_aggregates(monthly_rates):
    """计算季度和年度平均"""
    M = len(monthly_rates)
    
    # 季度平均
    quarters = []
    q_rates = []
    for q_start in range(0, M, 3):
        chunk = monthly_rates[q_start:q_start+3]
        if chunk:
            quarters.append(q_start//3 + 1)
            q_rates.append(sum(chunk) / len(chunk))
    
    # 年度平均
    years = []
    y_rates = []
    for y_start in range(0, M, 12):
        chunk = monthly_rates[y_start:y_start+12]
        if chunk:
            years.append(y_start//12 + 1)
            y_rates.append(sum(chunk) / len(chunk))
    
    return quarters, q_rates, years, y_rates

def compare_methods():
    """对比两种方法的差异"""
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    print("=" * 60)
    print("加载数据完成")
    print("=" * 60)
    
    # 检查数据结构
    print("\ndense_log 包含的键:", list(dense.keys()))
    
    actions = dense.get("actions", [])
    states = dense.get("states", [])
    
    print(f"\nactions 长度: {len(actions)}")
    print(f"states 长度: {len(states)}")
    
    if not actions:
        print("错误：没有 actions 数据！")
        return
    
    if not states:
        print("错误：没有 states 数据！")
        return
    
    # === 方法1: 基于 actions ===
    print("\n" + "=" * 60)
    print("方法1: 基于 actions (当月 l=0 的人数)")
    print("=" * 60)
    cnt1, rate1 = compute_unemployment_by_actions(actions)
    q1, qr1, y1, yr1 = compute_aggregates(rate1)
    
    print(f"\n年度失业率(%):", [round(x*100, 2) for x in yr1])
    
    # === 方法2: 基于 states ===
    print("\n" + "=" * 60)
    print("方法2: 基于 states (job='Unemployment' 的人数)")
    print("=" * 60)
    cnt2, rate2 = compute_unemployment_by_states(states)
    q2, qr2, y2, yr2 = compute_aggregates(rate2)
    
    print(f"\n年度失业率(%):", [round(x*100, 2) for x in yr2])
    
    # === 对比差异 ===
    print("\n" + "=" * 60)
    print("差异分析")
    print("=" * 60)
    
    # 月度差异
    print("\n前12个月的差异:")
    print("月份 | 方法1(actions) | 方法2(states) | 差值")
    print("-" * 55)
    for m in range(min(12, len(rate1))):
        diff = (rate1[m] - rate2[m]) * 100
        print(f"{m+1:3d}  | {rate1[m]*100:6.2f}%       | {rate2[m]*100:6.2f}%      | {diff:+6.2f}%")
    
    # 年度差异
    print("\n年度失业率对比:")
    print("年份 | 方法1(actions) | 方法2(states) | 差值")
    print("-" * 55)
    for i, (y, r1, r2) in enumerate(zip(y1, yr1, yr2)):
        diff = (r1 - r2) * 100
        print(f"{y:3d}  | {r1*100:6.2f}%       | {r2*100:6.2f}%      | {diff:+6.2f}%")
    
    # === 详细验证：检查具体agent ===
    print("\n" + "=" * 60)
    print("详细验证：检查第2-4个月的agent状态")
    print("=" * 60)
    
    for m in range(1, min(4, len(actions))):
        print(f"\n--- 月份 {m+1} ---")
        
        # 随机检查前5个agent
        for agent_id in ['0', '1', '2', '3', '4']:
            # 上个月的劳动决策
            prev_action = actions[m-1].get(agent_id, None)
            prev_labor = get_labor(prev_action) if prev_action is not None else -1
            
            # 本月的job状态
            curr_state = states[m].get(agent_id, {})
            curr_job = curr_state.get('endogenous', {}).get('job', 'N/A') if isinstance(curr_state, dict) else 'N/A'
            
            # 本月的劳动决策
            curr_action = actions[m].get(agent_id, None)
            curr_labor = get_labor(curr_action) if curr_action is not None else -1
            
            # 打印对比
            is_unemployed_by_action = "✓" if curr_labor == 0 else "✗"
            is_unemployed_by_state = "✓" if curr_job == 'Unemployment' else "✗"
            
            match = "✓" if (curr_labor == 0) == (curr_job == 'Unemployment') else "✗ 不匹配!"
            
            print(f"Agent {agent_id}: 上月labor={prev_labor}, 本月job='{curr_job[:15]}', 本月labor={curr_labor}")
            print(f"          失业判定: actions={is_unemployed_by_action}, states={is_unemployed_by_state} {match}")
    
    # === 保存结果 ===
    save_comparison_results(rate1, rate2, y1, yr1, yr2)
    
    return rate1, rate2, yr1, yr2

def save_comparison_results(rate1, rate2, years, yr1, yr2):
    """保存对比结果"""
    # 月度对比
    csv_path = os.path.join(OUT, "unemployment_comparison_monthly.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "rate_by_actions", "rate_by_states", "difference"])
        for m, (r1, r2) in enumerate(zip(rate1, rate2), start=1):
            w.writerow([m, r1, r2, r1 - r2])
    print(f"\n保存月度对比: {csv_path}")
    
    # 年度对比
    csv_path = os.path.join(OUT, "unemployment_comparison_yearly.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "rate_by_actions", "rate_by_states", "difference"])
        for y, r1, r2 in zip(years, yr1, yr2):
            w.writerow([y, r1, r2, r1 - r2])
    print(f"保存年度对比: {csv_path}")
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 年度对比
    axes[0].plot(years, [r*100 for r in yr1], marker='o', label='方法1: actions (l=0)', linewidth=2)
    axes[0].plot(years, [r*100 for r in yr2], marker='s', label='方法2: states (job=Unemployment)', linewidth=2)
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Unemployment Rate (%)', fontsize=12)
    axes[0].set_title('年度失业率对比', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 月度对比（前60个月）
    months = list(range(1, min(61, len(rate1)+1)))
    axes[1].plot(months, [r*100 for r in rate1[:60]], label='方法1: actions', linewidth=1, alpha=0.7)
    axes[1].plot(months, [r*100 for r in rate2[:60]], label='方法2: states', linewidth=1, alpha=0.7)
    axes[1].set_xlabel('Month', fontsize=12)
    axes[1].set_ylabel('Unemployment Rate (%)', fontsize=12)
    axes[1].set_title('月度失业率对比（前60个月）', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(OUT, "unemployment_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"保存对比图: {fig_path}")
    plt.close()

def compare_all_definitions():
    """对比三种失业率定义"""
    with open(P_DENSE, "rb") as f:
        dense = pkl.load(f)
    
    states = dense.get("states", [])
    actions = dense.get("actions", [])
    
    print("\n" + "="*70)
    print("三种失业率定义对比")
    print("="*70)
    
    # 定义1: 不工作率（actions中l=0）
    rates1 = []
    for m in range(len(actions)):
        cnt = sum(1 for k, v in actions[m].items() if k != 'p' and get_labor(v) == 0)
        rates1.append(cnt / 100)
    
    # 定义2: 标准失业率（states中job='Unemployment'，从第1月开始）
    rates2 = []
    for m in range(1, len(states)):  # ✅ 从第1个月开始
        unemployed = 0
        employed = 0
        for k, v in states[m].items():
            if k != 'p' and isinstance(v, dict):
                job = v.get('endogenous', {}).get('job')
                if job == 'Unemployment':
                    unemployed += 1
                else:
                    employed += 1
        
        labor_force = employed + unemployed
        rates2.append(unemployed / labor_force if labor_force > 0 else 0)
    
    # 定义3: 经济学标准失业率（与定义2相同的正确计算）
    rates3 = []
    for m in range(1, len(states)):
        unemployed = 0
        employed = 0
        for k, v in states[m].items():
            if k == 'p' or not isinstance(v, dict):
                continue
            job = v.get('endogenous', {}).get('job')
            if job == 'Unemployment':
                unemployed += 1
            else:
                employed += 1
        
        labor_force = employed + unemployed
        rates3.append(unemployed / labor_force if labor_force > 0 else 0)
    
    # 计算年度平均
    def yearly_avg(rates):
        years = []
        for y in range(0, min(len(rates), 240), 12):
            chunk = rates[y:y+12]
            if len(chunk) == 12:
                years.append(sum(chunk) / 12)
        return years
    
    y1 = yearly_avg(rates1)
    y2 = yearly_avg(rates2)
    y3 = yearly_avg(rates3)
    
    print("\n年度失业率对比（%）:")
    print("年份 | 定义1(不工作率) | 定义2(标准) | 定义3(经济学) | 论文预期")
    print("-" * 75)
    for i in range(min(len(y1), len(y2), len(y3))):
        print(f"{i+1:3d}  | {y1[i]*100:7.2f}        | {y2[i]*100:6.2f}    | {y3[i]*100:6.2f}      | 2-12")
    
    print("\n定义说明:")
    print("定义1 (不工作率): 统计actions中l=0的人（包含休假者，不推荐）")
    print("定义2 (标准失业率): 失业人数 / 劳动力（推荐！符合经济学定义）")
    print("定义3 (经济学标准): 与定义2相同（正确计算劳动力）")
    print("\n推荐使用: 定义2或定义3（两者应该相同）")
    
    # === 保存CSV ===
    csv_path = os.path.join(OUT, "unemployment_three_definitions_yearly.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "definition1_not_work", "definition2_standard", "definition3_economic"])
        for i in range(min(len(y1), len(y2), len(y3))):
            w.writerow([i+1, y1[i], y2[i], y3[i]])
    print(f"\n保存CSV: {csv_path}")
    
    # === 绘图 ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # ✅ 只保留这一行
    
    # 年度失业率图
    years_list = list(range(1, len(y2)+1))
    axes[0].plot(years_list, [r*100 for r in y2], marker='o', label='Unemployment Rate', 
                 linewidth=2, color='#2E86AB')
    axes[0].axhline(y=2, color='green', linestyle='--', alpha=0.5, 
                    label='Lower Bound (2%)')
    axes[0].axhline(y=12, color='red', linestyle='--', alpha=0.5, 
                    label='Upper Bound (12%)')
    axes[0].fill_between(years_list, 2, 12, alpha=0.1, color='green', 
                         label='Healthy Range')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Unemployment Rate (%)', fontsize=12)
    axes[0].set_title('Annual Unemployment Rate', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 30])
    
    # 月度失业率图（前60个月）
    months = list(range(1, min(61, len(rates2)+1)))
    axes[1].plot(months, [r*100 for r in rates2[:60]], label='Unemployment Rate', 
                 linewidth=1.5, alpha=0.8, color='#2E86AB')
    axes[1].axhline(y=2, color='green', linestyle='--', alpha=0.5)
    axes[1].axhline(y=12, color='red', linestyle='--', alpha=0.5)
    axes[1].fill_between(months, 2, 12, alpha=0.1, color='green')
    axes[1].set_xlabel('Month', fontsize=12)
    axes[1].set_ylabel('Unemployment Rate (%)', fontsize=12)
    axes[1].set_title('Monthly Unemployment Rate (First 60 Months)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(OUT, "unemployment_rate.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved: {fig_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("Files saved successfully!")
    print("="*70)
    
    return rates1, rates2, rates3, y1, y2, y3

if __name__ == "__main__":
    # Run unemployment rate analysis
    compare_all_definitions()

#     # 年度对比图
#     years_list = list(range(1, len(y1)+1))
#     axes[0].plot(years_list, [r*100 for r in y1], marker='o', label='定义1: 不工作率', linewidth=2)
#     axes[0].plot(years_list, [r*100 for r in y2], marker='s', label='定义2: 标准失业率', linewidth=2, linestyle='--')
#     axes[0].plot(years_list, [r*100 for r in y3], marker='^', label='定义3: 经济学标准', linewidth=2, linestyle=':')
#     axes[0].axhline(y=2, color='green', linestyle='--', alpha=0.5, label='论文预期下限(2%)')
#     axes[0].axhline(y=12, color='red', linestyle='--', alpha=0.5, label='论文预期上限(12%)')
#     axes[0].set_xlabel('Year', fontsize=12)
#     axes[0].set_ylabel('Unemployment Rate (%)', fontsize=12)
#     axes[0].set_title('年度失业率对比（三种定义）', fontsize=14, fontweight='bold')
#     axes[0].legend(fontsize=10)
#     axes[0].grid(True, alpha=0.3)
#     axes[0].set_ylim([0, 30])
    
#     # 月度对比图（前60个月）
#     months = list(range(1, min(61, len(rates2)+1)))
#     axes[1].plot(months, [r*100 for r in rates1[:60]], label='定义1: 不工作率', linewidth=1, alpha=0.8)
#     axes[1].plot(months, [r*100 for r in rates2[:60]], label='定义2: 标准失业率', linewidth=1, alpha=0.8, linestyle='--')
#     axes[1].plot(months, [r*100 for r in rates3[:60]], label='定义3: 经济学标准', linewidth=1, alpha=0.8, linestyle=':')
#     axes[1].axhline(y=2, color='green', linestyle='--', alpha=0.3)
#     axes[1].axhline(y=12, color='red', linestyle='--', alpha=0.3)
#     axes[1].set_xlabel('Month', fontsize=12)
#     axes[1].set_ylabel('Unemployment Rate (%)', fontsize=12)
#     axes[1].set_title('月度失业率对比（前60个月）', fontsize=14, fontweight='bold')
#     axes[1].legend(fontsize=10)
#     axes[1].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     fig_path = os.path.join(OUT, "unemployment_three_definitions.png")
#     plt.savefig(fig_path, dpi=300, bbox_inches='tight')
#     print(f"保存图表: {fig_path}")
#     plt.close()
    
#     print("\n" + "="*70)
#     print("文件保存完成！")
#     print("="*70)
    
#     return rates1, rates2, rates3, y1, y2, y3

# if __name__ == "__main__":
#     # 运行三种定义对比
#     compare_all_definitions()