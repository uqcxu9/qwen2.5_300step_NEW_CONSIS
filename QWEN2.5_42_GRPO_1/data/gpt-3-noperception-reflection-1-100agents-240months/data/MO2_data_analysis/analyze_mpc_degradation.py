#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MO2深度分析：失业示例质量 + MPC恶化原因"""

import pickle as pkl
import numpy as np
import pandas as pd
import os
# ========== 配置 ==========
MO2_DATA = r'C:\Users\徐瑞岑\OneDrive\PHD\experiment\QWEN2.5\QWEN_FEW_SHOT_ALL_1118\QWEN2.5_FEWSHOT_MO2\data\gpt-3-noperception-reflection-1-100agents-240months'
GOOD_CSV = os.path.join(MO2_DATA, 'good_decisions.csv')

print("="*60)
print("🔬 MO2深度分析：失业示例质量 + MPC恶化")
print("="*60)

# ========== 1. 加载Few-shot示例数据 ==========
print("\n📂 加载Few-shot示例...")
df = pd.read_csv(GOOD_CSV)
print(f"✅ 总示例数: {len(df)}")

employed = df[df['work_decision'] == 1.0]
unemployed = df[df['work_decision'] == 0.0]
print(f"   就业: {len(employed)} ({len(employed)/len(df)*100:.1f}%)")
print(f"   失业: {len(unemployed)} ({len(unemployed)/len(df)*100:.1f}%)")

# ========== 2. 失业示例质量分析 ==========
print("\n" + "="*60)
print("📊 Part 1: 失业示例质量检查")
print("="*60)

print("\n【失业示例的Score分布】")
print(f"均值: {unemployed['score'].mean():.2f}")
print(f"中位数: {unemployed['score'].median():.0f}")
print(f"范围: [{unemployed['score'].min():.0f}, {unemployed['score'].max():.0f}]")

print("\n【失业示例的MPC分布】")
unemployed_mpc = unemployed[unemployed['yearly_mpc'] != 0]['yearly_mpc']
if len(unemployed_mpc) > 0:
    print(f"样本数: {len(unemployed_mpc)}")
    print(f"均值: {unemployed_mpc.mean():.3f}")
    print(f"中位数: {unemployed_mpc.median():.3f}")
    print(f"范围: [{unemployed_mpc.min():.3f}, {unemployed_mpc.max():.3f}]")
    print(f"在[0.05,0.9]内: {((unemployed_mpc>=0.05)&(unemployed_mpc<=0.9)).sum()}/{len(unemployed_mpc)} ({((unemployed_mpc>=0.05)&(unemployed_mpc<=0.9)).sum()/len(unemployed_mpc)*100:.1f}%)")
    print(f"⚠️ 违约MPC(<0.05或>0.9): {((unemployed_mpc<0.05)|(unemployed_mpc>0.9)).sum()}/{len(unemployed_mpc)} ({((unemployed_mpc<0.05)|(unemployed_mpc>0.9)).sum()/len(unemployed_mpc)*100:.1f}%)")

print("\n【失业示例的储蓄率分布】")
unemployed_sr = []
for _, row in unemployed.iterrows():
    if row['curr_dpi'] > 50:
        sr = (row['curr_dpi'] - row['curr_consumption']) / row['curr_dpi']
        unemployed_sr.append(sr)
if unemployed_sr:
    print(f"样本数: {len(unemployed_sr)}")
    print(f"均值: {np.mean(unemployed_sr):.3f}")
    print(f"中位数: {np.median(unemployed_sr):.3f}")
    print(f"在[0.014,0.318]内: {sum((0.014<=sr<=0.318) for sr in unemployed_sr)}/{len(unemployed_sr)} ({sum((0.014<=sr<=0.318) for sr in unemployed_sr)/len(unemployed_sr)*100:.1f}%)")

print("\n【失业示例的收入/财富分布】")
print(f"平均收入: ${unemployed['curr_income'].mean():.2f}")
print(f"平均财富: ${unemployed['curr_wealth'].mean():.2f}")
print(f"收入范围: [${unemployed['curr_income'].min():.2f}, ${unemployed['curr_income'].max():.2f}]")
print(f"财富范围: [${unemployed['curr_wealth'].min():.2f}, ${unemployed['curr_wealth'].max():.2f}]")

print("\n【关键问题】")
low_income_unemployed = unemployed[unemployed['curr_income'] < 100]
print(f"低收入(<$100)失业: {len(low_income_unemployed)}/{len(unemployed)} ({len(low_income_unemployed)/len(unemployed)*100:.1f}%)")
print(f"→ 是否大多数失业是'被迫'（收入太低）而非'合理拒绝'？")

# ========== 3. MPC恶化原因分析 ==========
print("\n" + "="*60)
print("📊 Part 2: MPC恶化原因分析")
print("="*60)

print("\n【就业 vs 失业的MPC对比】")
employed_mpc = employed[employed['yearly_mpc'] != 0]['yearly_mpc']
unemployed_mpc = unemployed[unemployed['yearly_mpc'] != 0]['yearly_mpc']

if len(employed_mpc) > 0 and len(unemployed_mpc) > 0:
    print(f"\n就业示例MPC:")
    print(f"  均值: {employed_mpc.mean():.3f}")
    print(f"  违约率: {((employed_mpc<0.05)|(employed_mpc>0.9)).sum()/len(employed_mpc)*100:.1f}%")
    
    print(f"\n失业示例MPC:")
    print(f"  均值: {unemployed_mpc.mean():.3f}")
    print(f"  违约率: {((unemployed_mpc<0.05)|(unemployed_mpc>0.9)).sum()/len(unemployed_mpc)*100:.1f}%")
    
    print(f"\n⚠️ 关键发现:")
    if unemployed_mpc.mean() > employed_mpc.mean():
        print(f"   失业示例的MPC更高 ({unemployed_mpc.mean():.3f} vs {employed_mpc.mean():.3f})")
        print(f"   → 可能是失业时消费波动更大")
    
    unemployed_vr = ((unemployed_mpc<0.05)|(unemployed_mpc>0.9)).sum()/len(unemployed_mpc)*100
    employed_vr = ((employed_mpc<0.05)|(employed_mpc>0.9)).sum()/len(employed_mpc)*100
    if unemployed_vr > employed_vr:
        print(f"   失业示例的MPC违约率更高 ({unemployed_vr:.1f}% vs {employed_vr:.1f}%)")
        print(f"   → 失业示例本身质量不佳！")

print("\n【高分失业示例的MPC】")
unemployed_high_score = unemployed[unemployed['score'] >= 14]
if len(unemployed_high_score) > 0:
    high_score_mpc = unemployed_high_score[unemployed_high_score['yearly_mpc'] != 0]['yearly_mpc']
    if len(high_score_mpc) > 0:
        print(f"高分(≥14)失业示例数: {len(unemployed_high_score)}")
        print(f"MPC均值: {high_score_mpc.mean():.3f}")
        print(f"MPC违约率: {((high_score_mpc<0.05)|(high_score_mpc>0.9)).sum()/len(high_score_mpc)*100:.1f}%")
        print(f"→ 高分失业示例的MPC质量如何？")

# ========== 4. 实验结果数据加载 ==========
print("\n" + "="*60)
print("📊 Part 3: 实验结果验证")
print("="*60)

try:
    with open(f'{MO2_DATA}/dense_log.pkl', 'rb') as f:
        dense_log = pkl.load(f)
    
    states = dense_log['states']
    periodic_tax = dense_log['PeriodicTax']
    
    print("\n【实验中agent的实际行为】")
    
    # 计算年度MPC违约
    mpc_violations = 0
    mpc_total = 0
    
    for year in range(2, 21):
        year_start = (year - 1) * 12
        year_end = year * 12
        if year_end > len(states):
            break
        
        for agent_id in range(100):
            aid = str(agent_id)
            yearly_dpi_change = 0
            yearly_c_change = 0
            
            for t in range(year_start, year_end):
                if t == 0 or aid not in states[t]:
                    continue
                
                curr_dpi = states[t][aid]['income']['Coin'] + \
                          periodic_tax[t].get(aid, {}).get('lump_sum', 0) - \
                          periodic_tax[t].get(aid, {}).get('tax_paid', 0)
                prev_dpi = states[t-1][aid]['income']['Coin'] + \
                          periodic_tax[t-1].get(aid, {}).get('lump_sum', 0) - \
                          periodic_tax[t-1].get(aid, {}).get('tax_paid', 0)
                
                curr_c = states[t][aid]['consumption']['Coin']
                prev_c = states[t-1][aid]['consumption']['Coin']
                
                yearly_dpi_change += (curr_dpi - prev_dpi)
                yearly_c_change += (curr_c - prev_c)
            
            if abs(yearly_dpi_change) > 500:
                mpc_total += 1
                mpc = yearly_c_change / yearly_dpi_change
                if mpc < 0.05 or mpc > 0.9:
                    mpc_violations += 1
    
    print(f"实验中MPC违约率: {mpc_violations/mpc_total*100:.2f}% ({mpc_violations}/{mpc_total})")
    
    # 失业率分布
    print(f"\n【实验中失业率分布】")
    yearly_unemployment = []
    for year in range(2, 21):
        year_start = (year - 1) * 12
        year_end = year * 12
        if year_end > len(states):
            break
        
        year_unemp = []
        for t in range(year_start, year_end):
            unemployed = sum(1 for aid, state in states[t].items() 
                           if aid != "p" and isinstance(state, dict) 
                           and state.get("endogenous", {}).get("job") == "Unemployment")
            total = sum(1 for aid, state in states[t].items() 
                       if aid != "p" and isinstance(state, dict))
            year_unemp.append(unemployed / total * 100 if total > 0 else 0)
        yearly_unemployment.append(np.mean(year_unemp))
    
    print(f"平均年度失业率: {np.mean(yearly_unemployment):.2f}%")
    print(f"失业率范围: [{min(yearly_unemployment):.2f}%, {max(yearly_unemployment):.2f}%]")
    print(f"违约年份: {sum((u<3.5 or u>14.8) for u in yearly_unemployment)}/{len(yearly_unemployment)}")
    
except Exception as e:
    print(f"⚠️ 无法加载实验结果: {e}")

# ========== 5. 结论 ==========
print("\n" + "="*60)
print("💡 结论与建议")
print("="*60)

print("\n【问题诊断】")
print("1. 失业示例质量:")
if len(unemployed_mpc) > 0:
    unemployed_mpc_vr = ((unemployed_mpc<0.05)|(unemployed_mpc>0.9)).sum()/len(unemployed_mpc)*100
    if unemployed_mpc_vr > 50:
        print(f"   ❌ 失业示例MPC违约率{unemployed_mpc_vr:.1f}%，质量差")
    elif unemployed_mpc_vr > 30:
        print(f"   ⚠️ 失业示例MPC违约率{unemployed_mpc_vr:.1f}%，质量一般")
    else:
        print(f"   ✅ 失业示例MPC违约率{unemployed_mpc_vr:.1f}%，质量尚可")

print("\n2. 为什么MPC恶化:")
if len(employed_mpc) > 0 and len(unemployed_mpc) > 0:
    employed_mpc_vr = ((employed_mpc<0.05)|(employed_mpc>0.9)).sum()/len(employed_mpc)*100
    unemployed_mpc_vr = ((unemployed_mpc<0.05)|(unemployed_mpc>0.9)).sum()/len(unemployed_mpc)*100
    if unemployed_mpc_vr > employed_mpc_vr:
        print(f"   → 平衡采样引入了更多低质量MPC的失业示例")
        print(f"   → 就业示例MPC违约{employed_mpc_vr:.1f}% vs 失业示例{unemployed_mpc_vr:.1f}%")

print("\n【改进方向】")
if len(unemployed_mpc) > 0 and ((unemployed_mpc<0.05)|(unemployed_mpc>0.9)).sum()/len(unemployed_mpc) > 0.3:
    print("✅ 优先改进示例筛选逻辑:")
    print("   - 提高失业示例的MPC约束（只选MPC在[0.05,0.9]的）")
    print("   - 失业和就业分别筛选，确保两者质量一致")
    print("   - 调整score计算，给MPC更高权重")
else:
    print("⚠️ 示例质量尚可，问题可能在于:")
    print("   - 模型无法从示例中学到MPC约束")
    print("   - 需要更明确的prompt指导或考虑SFT")

print("\n✅ 分析完成！")