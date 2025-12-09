# -*- coding: utf-8 -*-
"""
consumption_income_ratio_test.py

测试 EconAgent 的消费-收入比稳定性约束

理性消费者应平滑消费。消费-收入比应在合理范围内且随时间保持稳定。

约束条件（基于美国消费模式数据）：
1. 范围约束: 0.50 ≤ R_{i,t} ≤ 0.90 (当收入 > 0 时)
2. 稳定性约束: Std(R_i) ≤ 0.20 (每个agent的时间序列标准差)

其中 R_{i,t} = C_{i,t} / Y_{i,t}
"""
import re
import os
import argparse
import pickle as pkl
import json
import csv
import statistics
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
# import matplotlib.pyplot as plt
from collections import deque

from collections import deque

def load_dialogs(data_path: str) -> Dict[int, Dict[str, str]]:
    """
    加载所有dialog文件并提取agent的完整对话历史
    
    Returns:
        dialogs: {t: {agent_id: full_conversation_text}}
    """
    dialogs = {}
    
    # 查找所有dialog文件
    for t in range(241):  # 0-240
        dialog_file = os.path.join(data_path, f"dialog_{t}.pkl")
        if os.path.exists(dialog_file):
            try:
                with open(dialog_file, 'rb') as f:
                    dialog_data = pkl.load(f)
                    
                    dialogs[t] = {}
                    
                    # ✅ 处理不同的数据格式
                    if isinstance(dialog_data, dict):
                        # 格式: {agent_id: deque([...])}
                        for aid, conversation in dialog_data.items():
                            if isinstance(conversation, (list, deque)):
                                # ✅ 提取完整对话历史
                                full_text = []
                                for msg in conversation:
                                    if isinstance(msg, dict):
                                        role = msg.get('role', 'unknown')
                                        content = msg.get('content', '')
                                        full_text.append(f"[{role}]: {content}")
                                    else:
                                        full_text.append(str(msg))
                                
                                dialogs[t][str(aid)] = "\n\n".join(full_text)
                            elif isinstance(conversation, str):
                                dialogs[t][str(aid)] = conversation
                            else:
                                dialogs[t][str(aid)] = str(conversation)
                                
                    elif isinstance(dialog_data, list):
                        # 格式: [agent0_conversation, agent1_conversation, ...]
                        for i, conversation in enumerate(dialog_data):
                            if isinstance(conversation, (list, deque)):
                                # ✅ 提取完整对话历史
                                full_text = []
                                for msg in conversation:
                                    if isinstance(msg, dict):
                                        role = msg.get('role', 'unknown')
                                        content = msg.get('content', '')
                                        full_text.append(f"[{role}]: {content}")
                                    else:
                                        full_text.append(str(msg))
                                
                                dialogs[t][str(i)] = "\n\n".join(full_text)
                            else:
                                dialogs[t][str(i)] = str(conversation)
                        
            except Exception as e:
                print(f"警告: 无法加载 dialog_{t}.pkl: {e}")
                dialogs[t] = {}
        else:
            dialogs[t] = {}
    
    return dialogs
def extract_expected_consumption_ratios(
    dense_log: Dict[str, Any],
    dialogs: Dict[int, Dict[str, str]]
) -> Dict[str, List[Tuple[int, float, float, float]]]:
    """
    从 dense_log['actions'] 提取期望消费-收入比
    
    Returns:
        agent_ratios: {agent_id: [(t, expected_consumption, disposable_income, expected_ratio)]}
    """
    states = dense_log['states']
    actions = dense_log['actions']
    periodic_tax_list = dense_log['PeriodicTax']
    
    agent_ratios = {}
    
    for t in range(len(states)):
        if t >= len(periodic_tax_list) or t >= len(actions):
            continue
        
        periodic_tax = periodic_tax_list[t]
        action_data = actions[t]  # 这个月所有 agent 的决策
        
        for aid in states[t].keys():
            if not aid.isdigit():
                continue
            
            agent_state = states[t][aid]
            
            # 计算可支配收入
            wage_income = agent_state['income'].get('Coin', 0)
            tax_info = periodic_tax.get(aid, {})
            tax_paid = tax_info.get('tax_paid', 0)
            lump_sum = tax_info.get('lump_sum', 0)
            disposable_income = wage_income - tax_paid + lump_sum
            
            if disposable_income <= 0:
                continue
            
            # ✅ 从 actions 中提取期望消费比例
            if aid in action_data:
                agent_action = action_data[aid]
                
                # ✅ actions 的格式: {'SimpleLabor': work, 'SimpleConsumption': consumption}
                if isinstance(agent_action, dict) and 'SimpleConsumption' in agent_action:
                    consumption_units = agent_action['SimpleConsumption']  # 0-50 的整数
                    
                    # 转换回比例 (0-50 → 0.00-1.00)
                    expected_ratio = consumption_units * 0.02
                    expected_consumption = expected_ratio * disposable_income
                    
                    if aid not in agent_ratios:
                        agent_ratios[aid] = []
                    
                    agent_ratios[aid].append((
                        t,
                        float(expected_consumption),
                        float(disposable_income),
                        float(expected_ratio)
                    ))
    
    return agent_ratios
def extract_consumption_income_ratios(
    dense_log: Dict[str, Any]
) -> Dict[str, List[Tuple[int, float, float, float]]]:
    """提取实际消费-可支配收入比"""
    
    states = dense_log['states']
    periodic_tax_list = dense_log['PeriodicTax']
    
    agent_ratios = {}
    
    for t in range(len(states)):
        if t >= len(periodic_tax_list):
            continue
        
        periodic_tax = periodic_tax_list[t]
        
        for aid in states[t].keys():
            if not aid.isdigit():
                continue
            
            agent_state = states[t][aid]
            
            # ✅ 提取实际消费（从 states）
            consumption = agent_state['consumption'].get('Coin', None)
            
            # 计算可支配收入
            wage_income = agent_state['income'].get('Coin', 0)
            tax_info = periodic_tax.get(aid, {})
            tax_paid = tax_info.get('tax_paid', 0)
            lump_sum = tax_info.get('lump_sum', 0)
            disposable_income = wage_income - tax_paid + lump_sum
            
            # 只跳过收入无效的情况
            if consumption is None or disposable_income <= 0:
                continue
            
            # ✅ 计算实际消费比例
            ratio = consumption / disposable_income
            
            if aid not in agent_ratios:
                agent_ratios[aid] = []
            
            agent_ratios[aid].append((
                t,
                float(consumption),
                float(disposable_income),
                float(ratio)
            ))
    
    return agent_ratios


def test_range_constraint(
    agent_ratios: Dict[str, List[Tuple[int, float, float, float]]],
    dialogs: Dict[int, Dict[str, str]],
    lower_bound: float = 0.50,
    upper_bound: float = 0.90
) -> Tuple[List[Dict], Dict]:
    """
    测试范围约束: 0.50 ≤ R_{i,t} ≤ 0.90
    
    Returns:
        violations: 违反列表 [{agent_id, t, ratio, violation_type}]
        summary: 统计摘要
    """
    violations = []
    all_ratios = []
    
    total_observations = 0
    in_range_count = 0
    too_low_count = 0
    too_high_count = 0
    
    for aid, records in agent_ratios.items():
        for t, consumption, income, ratio in records:
            total_observations += 1
            all_ratios.append(ratio)
            
            is_too_low = ratio < lower_bound
            is_too_high = ratio > upper_bound
            
            if is_too_low:
                too_low_count += 1
                response = dialogs.get(t, {}).get(aid, 'N/A')
                violations.append({
                    'agent_id': aid,
                    't': t,
                    'consumption': consumption,
                    'income': income,
                    'ratio': ratio,
                    'violation_type': 'too_low',
                    'deviation': lower_bound - ratio,
                    'response': response
                })
            elif is_too_high:
                too_high_count += 1
                response = dialogs.get(t, {}).get(aid, 'N/A')
                violations.append({
                    'agent_id': aid,
                    't': t,
                    'consumption': consumption,
                    'income': income,
                    'ratio': ratio,
                    'violation_type': 'too_high',
                    'deviation': ratio - upper_bound,
                    'response': response
                })
            else:
                in_range_count += 1
    
    summary = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'total_observations': total_observations,
        'in_range_count': in_range_count,
        'in_range_rate': in_range_count / total_observations if total_observations > 0 else 0,
        'too_low_count': too_low_count,
        'too_high_count': too_high_count,
        'violations_count': len(violations),
        'violation_rate': len(violations) / total_observations if total_observations > 0 else 0,
        'mean_ratio': np.mean(all_ratios) if all_ratios else 0,
        'median_ratio': np.median(all_ratios) if all_ratios else 0,
        'std_ratio': np.std(all_ratios, ddof=1) if all_ratios else 0,
        'min_ratio': min(all_ratios) if all_ratios else 0,
        'max_ratio': max(all_ratios) if all_ratios else 0
    }
    
    return violations, summary


def test_stability_constraint(
    agent_ratios: Dict[str, List[Tuple[int, float, float, float]]],
    max_std: float = 0.20
) -> Tuple[List[Dict], Dict]:
    """
    测试稳定性约束: Std(R_i) ≤ 0.20
    
    Returns:
        violations: 违反列表 [{agent_id, std, mean_ratio}]
        summary: 统计摘要
    """
    violations = []
    agent_stats = []
    
    for aid, records in agent_ratios.items():
        if len(records) < 2:
            continue
        
        ratios = [r[3] for r in records]  # 提取ratio
        
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios, ddof=1)  # 样本标准差
        
        agent_stats.append({
            'agent_id': aid,
            'n_observations': len(ratios),
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio
        })
        
        if std_ratio > max_std:
            violations.append({
                'agent_id': aid,
                'n_observations': len(ratios),
                'mean_ratio': mean_ratio,
                'std_ratio': std_ratio,
                'deviation': std_ratio - max_std
            })
    
    all_stds = [s['std_ratio'] for s in agent_stats]
    
    summary = {
        'max_std': max_std,
        'total_agents': len(agent_stats),
        'stable_agents': len(agent_stats) - len(violations),
        'stable_rate': (len(agent_stats) - len(violations)) / len(agent_stats) if agent_stats else 0,
        'violations_count': len(violations),
        'violation_rate': len(violations) / len(agent_stats) if agent_stats else 0,
        'mean_std': np.mean(all_stds) if all_stds else 0,
        'median_std': np.median(all_stds) if all_stds else 0,
        'min_std': min(all_stds) if all_stds else 0,
        'max_std_observed': max(all_stds) if all_stds else 0
    }
    
    return violations, summary, agent_stats


def print_ratio_report(
    range_violations: List[Dict],
    range_summary: Dict,
    stability_violations: List[Dict],
    stability_summary: Dict,
    top_k: int = 10
):
    """打印消费-收入比测试报告"""
    print("=" * 80)
    print("消费-收入比稳定性约束测试报告")
    print("=" * 80)
    
    print(f"\n理论背景：")
    print(f"  数据来源: Investopedia 美国消费模式数据")
    print(f"  理论基础: 理性消费者平滑消费行为")
    print(f"  MPC 典型范围: 0.2 - 0.9")
    
    print(f"\n约束条件：")
    print(f"  1. 范围约束: {range_summary['lower_bound']:.2f} ≤ R_{{i,t}} ≤ {range_summary['upper_bound']:.2f}")
    print(f"  2. 稳定性约束: Std(R_i) ≤ {stability_summary['max_std']:.2f}")
    
    print(f"\n" + "=" * 80)
    print("测试 1: 范围约束")
    print("=" * 80)
    
    print(f"\n总体统计：")
    print(f"  总观测数: {range_summary['total_observations']}")
    print(f"  符合范围: {range_summary['in_range_count']} ({range_summary['in_range_rate']:.2%})")
    print(f"  违反数: {range_summary['violations_count']} ({range_summary['violation_rate']:.2%})")
    print(f"    - 过低 (<{range_summary['lower_bound']}): {range_summary['too_low_count']}")
    print(f"    - 过高 (>{range_summary['upper_bound']}): {range_summary['too_high_count']}")
    
    print(f"\n消费-收入比统计：")
    print(f"  平均值: {range_summary['mean_ratio']:.4f}")
    print(f"  中位数: {range_summary['median_ratio']:.4f}")
    print(f"  标准差: {range_summary['std_ratio']:.4f}")
    print(f"  最小值: {range_summary['min_ratio']:.4f}")
    print(f"  最大值: {range_summary['max_ratio']:.4f}")
    
    # 判断范围约束
    if range_summary['violations_count'] == 0:
        print(f"\n✅ 范围约束通过！所有观测都在合理范围内！")
    else:
        print(f"\n⚠️  范围约束: 发现 {range_summary['violations_count']} 个违反")
        
        if range_summary['violation_rate'] < 0.05:
            print(f"   轻微违反 (违反率 < 5%)")
        elif range_summary['violation_rate'] < 0.20:
            print(f"   中度违反 (违反率 < 20%)")
        else:
            print(f"   严重违反 (违反率 ≥ 20%)")
        
        # 显示最严重的违反
        print(f"\n前 {min(top_k, len(range_violations))} 个最严重的范围违反:")
        print("-" * 80)
        
        sorted_violations = sorted(range_violations, key=lambda x: abs(x['deviation']), reverse=True)[:top_k]
        
        for rank, v in enumerate(sorted_violations, 1):
            violation_desc = "过低（消费不足）" if v['violation_type'] == 'too_low' else "过高（过度消费）"
            
            print(f"\n第 {rank} 名: [{violation_desc}]")
            print(f"  Agent: {v['agent_id']}, 时间步: {v['t']}")
            print(f"  消费: {v['consumption']:.2f}, 收入: {v['income']:.2f}")
            print(f"  比率: {v['ratio']:.4f}")
            print(f"  偏离: {abs(v['deviation']):.4f}")
            if 'response' in v and v['response'] != 'N/A':
                response_text = v['response']
                print(f"  Agent完整响应:")
                print(f"  {'-'*76}")
                # 缩进显示response
                for line in response_text.split('\n'):
                    print(f"  {line}")
                print(f"  {'-'*76}")
            else:
                print(f"  Agent响应: [无响应数据 - 此时间步未保存dialog]")                
       
    
    print(f"\n" + "=" * 80)
    print("测试 2: 稳定性约束")
    print("=" * 80)
    
    print(f"\n总体统计：")
    print(f"  总agent数: {stability_summary['total_agents']}")
    print(f"  稳定agent数: {stability_summary['stable_agents']} ({stability_summary['stable_rate']:.2%})")
    print(f"  不稳定agent数: {stability_summary['violations_count']} ({stability_summary['violation_rate']:.2%})")
    
    print(f"\n时间序列标准差统计：")
    print(f"  平均标准差: {stability_summary['mean_std']:.4f}")
    print(f"  中位数: {stability_summary['median_std']:.4f}")
    print(f"  最小值: {stability_summary['min_std']:.4f}")
    print(f"  最大值: {stability_summary['max_std_observed']:.4f}")
    
    # 判断稳定性约束
    if stability_summary['violations_count'] == 0:
        print(f"\n✅ 稳定性约束通过！所有agent的消费-收入比都稳定！")
    else:
        print(f"\n⚠️  稳定性约束: 发现 {stability_summary['violations_count']} 个违反")
        
        if stability_summary['violation_rate'] < 0.05:
            print(f"   轻微违反 (违反率 < 5%)")
        elif stability_summary['violation_rate'] < 0.20:
            print(f"   中度违反 (违反率 < 20%)")
        else:
            print(f"   严重违反 (违反率 ≥ 20%)")
        
        # 显示最不稳定的agent
        print(f"\n前 {min(top_k, len(stability_violations))} 个最不稳定的agent:")
        print("-" * 80)
        
        sorted_violations = sorted(stability_violations, key=lambda x: x['std_ratio'], reverse=True)[:top_k]
        
        for rank, v in enumerate(sorted_violations, 1):
            print(f"\n第 {rank} 名:")
            print(f"  Agent: {v['agent_id']}")
            print(f"  观测数: {v['n_observations']}")
            print(f"  平均比率: {v['mean_ratio']:.4f}")
            print(f"  标准差: {v['std_ratio']:.4f} (阈值: {stability_summary['max_std']:.2f})")
            print(f"  超出: {v['deviation']:.4f}")
    
    print(f"\n" + "=" * 80)
    print("综合评估")
    print("=" * 80)
    
    range_pass = range_summary['violation_rate'] < 0.05
    stability_pass = stability_summary['violation_rate'] < 0.05
    
    if range_pass and stability_pass:
        print(f"\n✅ 两项约束均通过！")
        print(f"   消费-收入比在合理范围内且随时间稳定")
    elif range_pass:
        print(f"\n⚠️  范围约束通过，但稳定性约束未通过")
    elif stability_pass:
        print(f"\n⚠️  稳定性约束通过，但范围约束未通过")
    else:
        print(f"\n❌ 两项约束均未通过")
    
    print("\n" + "=" * 80)


def plot_consumption_income_ratios(
    agent_ratios: Dict[str, List[Tuple[int, float, float, float]]],
    range_summary: Dict,
    stability_summary: Dict,
    agent_stats: List[Dict],
    save_path: Optional[str] = None
):
    """绘制消费-收入比可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 提取所有比率
    all_ratios = []
    for records in agent_ratios.values():
        all_ratios.extend([r[3] for r in records])
    
    # 子图1: 消费-收入比分布（直方图）
    ax1 = axes[0, 0]
    ax1.hist(all_ratios, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=range_summary['lower_bound'], color='green', linestyle='--', 
                linewidth=2, label=f"下界 ({range_summary['lower_bound']})")
    ax1.axvline(x=range_summary['upper_bound'], color='red', linestyle='--', 
                linewidth=2, label=f"上界 ({range_summary['upper_bound']})")
    ax1.axvline(x=range_summary['mean_ratio'], color='orange', linestyle='-', 
                linewidth=2, label=f"平均值 ({range_summary['mean_ratio']:.2f})")
    ax1.set_xlabel('消费-收入比', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('消费-收入比分布', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 时间序列标准差分布
    ax2 = axes[0, 1]
    stds = [s['std_ratio'] for s in agent_stats]
    ax2.hist(stds, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(x=stability_summary['max_std'], color='red', linestyle='--', 
                linewidth=2, label=f"阈值 ({stability_summary['max_std']})")
    ax2.axvline(x=stability_summary['mean_std'], color='blue', linestyle='-', 
                linewidth=2, label=f"平均值 ({stability_summary['mean_std']:.2f})")
    ax2.set_xlabel('标准差', fontsize=12)
    ax2.set_ylabel('Agent数', fontsize=12)
    ax2.set_title('时间序列标准差分布', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 随机选择5个agent的时间序列
    ax3 = axes[1, 0]
    sample_agents = list(agent_ratios.keys())[:5]
    for aid in sample_agents:
        records = agent_ratios[aid]
        t_values = [r[0] for r in records]
        ratios = [r[3] for r in records]
        ax3.plot(t_values, ratios, alpha=0.6, label=f'Agent {aid}')
    
    ax3.axhline(y=range_summary['lower_bound'], color='green', linestyle='--', 
                linewidth=1, alpha=0.5)
    ax3.axhline(y=range_summary['upper_bound'], color='red', linestyle='--', 
                linewidth=1, alpha=0.5)
    ax3.fill_between([0, max([r[0] for records in agent_ratios.values() for r in records])],
                     range_summary['lower_bound'], range_summary['upper_bound'],
                     alpha=0.1, color='green')
    ax3.set_xlabel('时间步', fontsize=12)
    ax3.set_ylabel('消费-收入比', fontsize=12)
    ax3.set_title('样本Agent时间序列', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 平均比率 vs 标准差散点图
    ax4 = axes[1, 1]
    means = [s['mean_ratio'] for s in agent_stats]
    stds = [s['std_ratio'] for s in agent_stats]
    ax4.scatter(means, stds, alpha=0.5, s=20, color='purple')
    ax4.axhline(y=stability_summary['max_std'], color='red', linestyle='--', 
                linewidth=2, label=f"稳定性阈值")
    ax4.axvline(x=range_summary['lower_bound'], color='green', linestyle='--', 
                linewidth=1, alpha=0.5)
    ax4.axvline(x=range_summary['upper_bound'], color='red', linestyle='--', 
                linewidth=1, alpha=0.5)
    ax4.set_xlabel('平均消费-收入比', fontsize=12)
    ax4.set_ylabel('标准差', fontsize=12)
    ax4.set_title('平均值 vs 稳定性', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 设置字体以支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
def main():
    parser = argparse.ArgumentParser(
        description="EconAgent 消费-收入比稳定性约束测试"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="数据目录路径"
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default="dense_log.pkl",
        help="pkl文件名"
    )
    parser.add_argument(
        "--lower_bound",
        type=float,
        default=0.50,
        help="消费-收入比下界"
    )
    parser.add_argument(
        "--upper_bound",
        type=float,
        default=0.90,
        help="消费-收入比上界"
    )
    parser.add_argument(
        "--max_std",
        type=float,
        default=0.20,
        help="标准差上限"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="显示前K个违反"
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="保存结果到JSON文件"
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="保存详细数据到CSV"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="保存图表到文件"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    pkl_path = os.path.join(args.data_path, args.pickle)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到文件：{pkl_path}")
    
    print("正在加载数据...")
    with open(pkl_path, 'rb') as f:
        dense_log = pkl.load(f)
    print("数据加载完成！\n")
    
    print("正在加载agent响应数据...")
    dialogs = load_dialogs(args.data_path)
    print(f"已加载 {len([d for d in dialogs.values() if d])} 个时间步的响应数据\n")
    
    # 提取实际消费-收入比
    print("提取实际消费-收入比数据...")
    actual_ratios = extract_consumption_income_ratios(dense_log)
    actual_obs = sum(len(records) for records in actual_ratios.values())
    print(f"已提取 {len(actual_ratios)} 个agent的实际消费数据，共 {actual_obs} 个观测\n")
    
    # 提取期望消费-收入比
    print("提取期望消费-收入比数据（从Agent响应中）...")
    expected_ratios = extract_expected_consumption_ratios(dense_log, dialogs)
    expected_obs = sum(len(records) for records in expected_ratios.values())
    print(f"已提取 {len(expected_ratios)} 个agent的期望消费数据，共 {expected_obs} 个观测\n")
    
    # ============ 测试1：期望消费（Agent决策质量） ============
    print("=" * 80)
    print("测试 Agent 期望消费（决策质量）")
    print("=" * 80)

    print("\n测试范围约束（期望）...")
    expected_range_violations, expected_range_summary = test_range_constraint(
        expected_ratios,
        dialogs,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound
    )
    print("期望范围约束测试完成！\n")

    print("测试稳定性约束（期望）...")
    expected_stability_violations, expected_stability_summary, expected_agent_stats = test_stability_constraint(
        expected_ratios,
        max_std=args.max_std
    )
    print("期望稳定性约束测试完成！\n")

    # ============ 测试2：实际消费（系统整体表现） ============
    print("=" * 80)
    print("测试 Agent 实际消费（受市场限制）")
    print("=" * 80)

    print("\n测试范围约束（实际）...")
    actual_range_violations, actual_range_summary = test_range_constraint(
        actual_ratios,
        dialogs,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound
    )
    print("实际范围约束测试完成！\n")

    print("测试稳定性约束（实际）...")
    actual_stability_violations, actual_stability_summary, actual_agent_stats = test_stability_constraint(
        actual_ratios,
        max_std=args.max_std
    )
    print("实际稳定性约束测试完成！\n")

    # ============ 打印对比报告 ============
    print("\n" + "=" * 80)
    print("双重测试对比报告")
    print("=" * 80)

    print("\n【期望消费】Agent 决策质量测试：")
    print("-" * 80)
    print_ratio_report(
        expected_range_violations, expected_range_summary,
        expected_stability_violations, expected_stability_summary,
        top_k=args.top_k
    )

    print("\n" + "=" * 80)
    print("\n【实际消费】系统整体表现测试（受市场限制）：")
    print("-" * 80)
    print_ratio_report(
        actual_range_violations, actual_range_summary,
        actual_stability_violations, actual_stability_summary,
        top_k=args.top_k
    )

    # ============ 对比分析 ============
    print("\n" + "=" * 80)
    print("市场影响分析")
    print("=" * 80)

    print("\n范围约束违反对比：")
    print(f"  期望消费违反: {len(expected_range_violations)} ({expected_range_summary['violation_rate']:.2%})")
    print(f"  实际消费违反: {len(actual_range_violations)} ({actual_range_summary['violation_rate']:.2%})")
    print(f"  差异: {len(actual_range_violations) - len(expected_range_violations)} 个")

    if len(actual_range_violations) > len(expected_range_violations):
        print(f"  ⚠️  市场限制导致 {len(actual_range_violations) - len(expected_range_violations)} 个额外违反")
    elif len(actual_range_violations) < len(expected_range_violations):
        print(f"  ✅ 市场限制减少了 {len(expected_range_violations) - len(actual_range_violations)} 个违反")
    else:
        print(f"  ✅ 市场影响可忽略")

    print("\n稳定性约束违反对比：")
    print(f"  期望消费违反: {len(expected_stability_violations)} ({expected_stability_summary['violation_rate']:.2%})")
    print(f"  实际消费违反: {len(actual_stability_violations)} ({actual_stability_summary['violation_rate']:.2%})")
    print(f"  差异: {len(actual_stability_violations) - len(expected_stability_violations)} 个")

    print("\n" + "=" * 80)
    
    # 绘制图表
    if args.plot:
        # 期望消费图
        expected_plot_path = args.plot.replace('.png', '_expected.png')
        plot_consumption_income_ratios(
            expected_ratios, expected_range_summary, 
            expected_stability_summary, expected_agent_stats,
            save_path=os.path.join(args.data_path, expected_plot_path)
        )
        
        # 实际消费图
        actual_plot_path = args.plot.replace('.png', '_actual.png')
        plot_consumption_income_ratios(
            actual_ratios, actual_range_summary, 
            actual_stability_summary, actual_agent_stats,
            save_path=os.path.join(args.data_path, actual_plot_path)
        )
    
    # 保存结果到JSON
    if args.save_json:
        results = {
            'expected_consumption': {
                'range_constraint': {
                    'summary': expected_range_summary,
                    'violations_count': len(expected_range_violations),
                    'top_violations': sorted(expected_range_violations, 
                                            key=lambda x: abs(x['deviation']), 
                                            reverse=True)[:args.top_k]
                },
                'stability_constraint': {
                    'summary': expected_stability_summary,
                    'violations_count': len(expected_stability_violations),
                    'top_violations': sorted(expected_stability_violations,
                                            key=lambda x: x['std_ratio'],
                                            reverse=True)[:args.top_k]
                }
            },
            'actual_consumption': {
                'range_constraint': {
                    'summary': actual_range_summary,
                    'violations_count': len(actual_range_violations),
                    'top_violations': sorted(actual_range_violations, 
                                            key=lambda x: abs(x['deviation']), 
                                            reverse=True)[:args.top_k]
                },
                'stability_constraint': {
                    'summary': actual_stability_summary,
                    'violations_count': len(actual_stability_violations),
                    'top_violations': sorted(actual_stability_violations,
                                            key=lambda x: x['std_ratio'],
                                            reverse=True)[:args.top_k]
                }
            },
            'market_impact': {
                'range_violation_diff': len(actual_range_violations) - len(expected_range_violations),
                'stability_violation_diff': len(actual_stability_violations) - len(expected_stability_violations)
            }
        }
        
        json_path = os.path.join(args.data_path, args.save_json)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {json_path}")
    
    # 保存详细数据到CSV
    if args.save_csv:
        # 保存期望消费数据
        expected_csv = args.save_csv.replace('.csv', '_expected.csv')
        csv_path = os.path.join(args.data_path, expected_csv)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['agent_id', 't', 'expected_consumption', 'income', 'expected_ratio'])
            for aid, records in expected_ratios.items():
                for t, c, y, r in records:
                    writer.writerow([aid, t, c, y, r])
        print(f"✅ 期望消费数据已保存到: {csv_path}")
        
        # 保存实际消费数据
        actual_csv = args.save_csv.replace('.csv', '_actual.csv')
        csv_path = os.path.join(args.data_path, actual_csv)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['agent_id', 't', 'actual_consumption', 'income', 'actual_ratio'])
            for aid, records in actual_ratios.items():
                for t, c, y, r in records:
                    writer.writerow([aid, t, c, y, r])
        print(f"✅ 实际消费数据已保存到: {csv_path}")


if __name__ == "__main__":
    main()