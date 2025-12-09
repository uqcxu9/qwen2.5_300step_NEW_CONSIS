# -*- coding: utf-8 -*-
"""
wage_redistribution_test.py

测试 EconAgent 的税收再分配机制

约束条件：
1. 总税收 = 总转移支付（每个时间步）
2. 累进税制：有效税率随收入增加而增加（Spearman相关系数 > 0）
"""

import os
import argparse
import pickle as pkl
import json
import csv
import statistics
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import spearmanr
import numpy as np


def is_agent_id(key: str) -> bool:
    """判断key是否是agent ID（数字字符串）"""
    try:
        int(key)
        return True
    except (ValueError, TypeError):
        return False


def test_redistribution_balance(
    dense_log: Dict[str, Any],
    tolerance: float = 1.0
) -> Tuple[List[Dict], Dict]:
    """
    测试税收再分配的平衡性
    
    验证：总税收 = 总转移支付
    """
    periodic_tax = dense_log.get('PeriodicTax', [])
    
    if not periodic_tax:
        raise ValueError("PeriodicTax 数据不存在")
    
    violations = []
    all_errors = []
    total_checks = 0
    
    T = len(periodic_tax)
    
    for t in range(T):
        # 获取所有agent（只要数字ID）
        agent_ids = [aid for aid in periodic_tax[t].keys() if is_agent_id(aid)]
        
        # 计算总税收和总转移支付
        total_tax = sum(
            float(periodic_tax[t][aid]['tax_paid'])
            for aid in agent_ids
        )
        
        total_lump_sum = sum(
            float(periodic_tax[t][aid]['lump_sum'])
            for aid in agent_ids
        )
        
        # 计算误差
        error = abs(total_tax - total_lump_sum)
        all_errors.append(error)
        total_checks += 1
        
        if error > tolerance:
            violations.append({
                't': t,
                'year': t // 12 + 1,
                'month': t % 12 + 1,
                'total_tax': total_tax,
                'total_lump_sum': total_lump_sum,
                'difference': total_tax - total_lump_sum,
                'error': error,
                'num_agents': len(agent_ids)
            })
    
    # 统计摘要
    summary = {
        'total_checks': total_checks,
        'violations_count': len(violations),
        'violation_rate': len(violations) / total_checks if total_checks > 0 else 0,
        'max_error': max(all_errors) if all_errors else 0.0,
        'mean_error': statistics.mean(all_errors) if all_errors else 0.0,
        'median_error': statistics.median(all_errors) if all_errors else 0.0,
        'mean_error_violations': statistics.mean([v['error'] for v in violations]) if violations else 0.0
    }
    
    return violations, summary


def test_progressive_taxation(
    dense_log: Dict[str, Any]
) -> Tuple[List[Dict], Dict]:
    """
    测试累进税制
    
    验证：有效税率随收入增加而增加
    使用 Spearman 相关系数
    """
    periodic_tax = dense_log.get('PeriodicTax', [])
    
    if not periodic_tax:
        raise ValueError("PeriodicTax 数据不存在")
    
    correlations = []
    T = len(periodic_tax)
    
    for t in range(T):
        # 获取所有agent（只要数字ID）
        agent_ids = [aid for aid in periodic_tax[t].keys() if is_agent_id(aid)]
        
        # 提取数据
        incomes = []
        effective_rates = []
        
        for aid in agent_ids:
            income = float(periodic_tax[t][aid]['income'])
            effective_rate = float(periodic_tax[t][aid]['effective_rate'])
            
            # 只考虑有收入的agent
            if income > 0:
                incomes.append(income)
                effective_rates.append(effective_rate)
        
        # 计算 Spearman 相关系数
        if len(incomes) >= 3:  # 至少需要3个数据点
            rho, p_value = spearmanr(incomes, effective_rates)
            
            correlations.append({
                't': t,
                'year': t // 12 + 1,
                'month': t % 12 + 1,
                'spearman_rho': float(rho),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05),  # ← 转换为bool
                'is_positive': bool(rho > 0),            # ← 转换为bool
                'is_progressive': bool((rho > 0) and (p_value < 0.05)),  # ← 转换为bool
                'num_samples': len(incomes),
                'mean_income': statistics.mean(incomes),
                'mean_effective_rate': statistics.mean(effective_rates),
                'min_effective_rate': min(effective_rates),
                'max_effective_rate': max(effective_rates)
            })
    
    # 统计摘要
    progressive_count = sum(1 for c in correlations if c['is_progressive'])
    positive_count = sum(1 for c in correlations if c['is_positive'])
    significant_count = sum(1 for c in correlations if c['is_significant'])
    
    summary = {
        'total_periods': len(correlations),
        'progressive_count': progressive_count,
        'progressive_rate': progressive_count / len(correlations) if correlations else 0,
        'positive_correlation_count': positive_count,
        'positive_correlation_rate': positive_count / len(correlations) if correlations else 0,
        'significant_correlation_count': significant_count,
        'significant_correlation_rate': significant_count / len(correlations) if correlations else 0,
        'mean_rho': statistics.mean([c['spearman_rho'] for c in correlations]) if correlations else 0,
        'median_rho': statistics.median([c['spearman_rho'] for c in correlations]) if correlations else 0,
        'mean_p_value': statistics.mean([c['p_value'] for c in correlations]) if correlations else 0,
        'min_rho': min([c['spearman_rho'] for c in correlations]) if correlations else 0,
        'max_rho': max([c['spearman_rho'] for c in correlations]) if correlations else 0
    }
    
    return correlations, summary


def print_redistribution_report(
    violations: List[Dict],
    summary: Dict,
    top_k: int = 10
):
    """打印税收再分配平衡测试报告"""
    print("=" * 80)
    print("税收再分配平衡测试报告")
    print("=" * 80)
    
    print(f"\n约束条件: ∑税收 = ∑转移支付")
    print(f"总检查次数: {summary['total_checks']:,}")
    print(f"违反次数: {summary['violations_count']:,}")
    print(f"违反率: {summary['violation_rate']:.6%}")
    print(f"最大误差: ${summary['max_error']:.2f}")
    print(f"平均误差 (所有): ${summary['mean_error']:.6f}")
    print(f"中位数误差 (所有): ${summary['median_error']:.6f}")
    
    if summary['violations_count'] == 0:
        print("\n✅ 测试通过！所有时间步的税收再分配均完美平衡！")
    else:
        print(f"\n⚠️  发现 {summary['violations_count']} 个不平衡情况")
        print(f"平均误差 (仅违反): ${summary['mean_error_violations']:.2f}")
        
        print(f"\n前 {min(top_k, len(violations))} 个最严重的不平衡:")
        print("-" * 80)
        
        sorted_violations = sorted(violations, key=lambda x: x['error'], reverse=True)[:top_k]
        
        for rank, v in enumerate(sorted_violations, 1):
            print(f"\n第 {rank} 名:")
            print(f"  时间: 第{v['year']}年第{v['month']}月 (t={v['t']})")
            print(f"  总税收: ${v['total_tax']:,.2f}")
            print(f"  总转移支付: ${v['total_lump_sum']:,.2f}")
            print(f"  差额: ${v['difference']:,.2f}")
            print(f"  误差: ${v['error']:,.2f}")
            print(f"  agent数量: {v['num_agents']}")
    
    print("\n" + "=" * 80)


def print_progressive_report(
    correlations: List[Dict],
    summary: Dict,
    show_failed: bool = True,
    top_k: int = 10
):
    """打印累进税制测试报告"""
    print("=" * 80)
    print("累进税制测试报告")
    print("=" * 80)
    
    print(f"\n约束条件: 有效税率 ↑ 当 收入 ↑")
    print(f"统计方法: Spearman 相关系数 (ρ > 0, p < 0.05)")
    
    print(f"\n总时间步数: {summary['total_periods']:,}")
    print(f"累进税制时期数: {summary['progressive_count']:,}")
    print(f"累进税制比例: {summary['progressive_rate']:.2%}")
    
    print(f"\n详细统计:")
    print(f"  正相关时期: {summary['positive_correlation_count']:,} ({summary['positive_correlation_rate']:.1%})")
    print(f"  显著相关时期: {summary['significant_correlation_count']:,} ({summary['significant_correlation_rate']:.1%})")
    
    print(f"\nSpearman 相关系数 (ρ) 统计:")
    print(f"  平均 ρ: {summary['mean_rho']:.6f}")
    print(f"  中位数 ρ: {summary['median_rho']:.6f}")
    print(f"  最小 ρ: {summary['min_rho']:.6f}")
    print(f"  最大 ρ: {summary['max_rho']:.6f}")
    print(f"  平均 p值: {summary['mean_p_value']:.6f}")
    
    if summary['progressive_rate'] >= 0.95:
        print("\n✅ 测试通过！绝大多数时期 (≥95%) 表现出累进税制！")
    elif summary['progressive_rate'] >= 0.80:
        print("\n⚠️  大部分时期 (≥80%) 表现出累进税制，但有少数例外")
    else:
        print(f"\n❌ 测试失败！仅 {summary['progressive_rate']:.1%} 的时期满足累进税制")
    
    # 显示不满足累进税制的时期
    if show_failed:
        failed = [c for c in correlations if not c['is_progressive']]
        
        if failed:
            print(f"\n前 {min(top_k, len(failed))} 个不满足累进税制的时期:")
            print("-" * 80)
            
            # 按 rho 从小到大排序（最不累进的）
            sorted_failed = sorted(failed, key=lambda x: x['spearman_rho'])[:top_k]
            
            for rank, c in enumerate(sorted_failed, 1):
                status = []
                if not c['is_positive']:
                    status.append("负相关")
                if not c['is_significant']:
                    status.append("不显著")
                status_str = ", ".join(status) if status else "未知原因"
                
                print(f"\n第 {rank} 名: [{status_str}]")
                print(f"  时间: 第{c['year']}年第{c['month']}月 (t={c['t']})")
                print(f"  Spearman ρ: {c['spearman_rho']:.6f}")
                print(f"  p值: {c['p_value']:.6f}")
                print(f"  样本数: {c['num_samples']}")
                print(f"  有效税率范围: {c['min_effective_rate']:.4f} - {c['max_effective_rate']:.4f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="EconAgent 工资再分配机制测试"
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
        "--tolerance",
        type=float,
        default=1.0,
        help="再分配平衡的误差容差（美元）"
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
        "--save_csv_balance",
        type=str,
        default=None,
        help="保存再分配平衡违反到CSV"
    )
    parser.add_argument(
        "--save_csv_progressive",
        type=str,
        default=None,
        help="保存累进税制数据到CSV"
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
    
    # 测试1: 税收再分配平衡
    print("测试 1/2: 税收再分配平衡")
    print("=" * 80)
    violations_balance, summary_balance = test_redistribution_balance(
        dense_log,
        tolerance=args.tolerance
    )
    print_redistribution_report(violations_balance, summary_balance, top_k=args.top_k)
    
    # 测试2: 累进税制
    print("\n测试 2/2: 累进税制")
    print("=" * 80)
    correlations, summary_progressive = test_progressive_taxation(dense_log)
    print_progressive_report(correlations, summary_progressive, top_k=args.top_k)
    
    # 保存结果
    if args.save_json:
        results = {
            'redistribution_balance': {
                'summary': summary_balance,
                'violations': violations_balance
            },
            'progressive_taxation': {
                'summary': summary_progressive,
                'correlations': correlations
            }
        }
        json_path = os.path.join(args.data_path, args.save_json)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {json_path}")
    
    if args.save_csv_balance and violations_balance:
        csv_path = os.path.join(args.data_path, args.save_csv_balance)
        fieldnames = ['t', 'year', 'month', 'total_tax', 'total_lump_sum', 
                     'difference', 'error', 'num_agents']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for v in violations_balance:
                writer.writerow({k: v.get(k, '') for k in fieldnames})
        print(f"✅ 再分配平衡违反已保存到: {csv_path}")
    
    if args.save_csv_progressive:
        csv_path = os.path.join(args.data_path, args.save_csv_progressive)
        fieldnames = ['t', 'year', 'month', 'spearman_rho', 'p_value',
                     'is_significant', 'is_positive', 'is_progressive',
                     'num_samples', 'mean_income', 'mean_effective_rate']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in correlations:
                writer.writerow({k: c.get(k, '') for k in fieldnames})
        print(f"✅ 累进税制数据已保存到: {csv_path}")


if __name__ == "__main__":
    main()