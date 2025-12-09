# -*- coding: utf-8 -*-
"""
budget_constraint_final_v3.py

EconAgent 预算约束测试（最终版 v3 - 排除第1年）

跳过第1年末的结算（t=11→12），因为第1年的利率设定与后续年份不同
"""

import os
import argparse
import pickle as pkl
import json
import csv
import statistics
from typing import Dict, List, Tuple, Optional, Any


def get_interest_rate(t: int, world: List[Dict]) -> Optional[float]:
    """
    获取 t→t+1 过渡时的利率
    
    Returns:
        None: 不是年初或者是第1年末
        float: 利率值
    """
    if (t + 1) % 12 != 0:
        return None
    
    year = (t + 1) // 12
    
    # 跳过第1年末
    if year == 1:
        return None
    
    # 其他年份：使用上一年初设定的利率
    rate_t = (year - 1) * 12
    if rate_t >= len(world):
        return None
    
    rate = world[rate_t].get('Interest Rate', None)
    return float(rate) if rate is not None else None


def test_budget_constraint_final(
    dense_log: Dict[str, Any],
    tolerance: float = 1.0,
    skip_year1: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    测试预算约束（最终版本）
    
    Args:
        dense_log: 完整日志数据
        tolerance: 误差容差
        skip_year1: 是否跳过第1年末的检查
    """
    states = dense_log.get('states', [])
    periodic_tax = dense_log.get('PeriodicTax', [])
    world = dense_log.get('world', [])
    
    if len(states) < 2 or len(periodic_tax) < 1:
        raise ValueError("数据不足，无法进行测试")
    
    violations = []
    all_errors = []
    total_checks = 0
    skipped_year1 = 0
    interest_periods = 0
    
    T = min(len(states), len(periodic_tax))
    
    first_agents = list(states[0].keys())
    agent_ids = [a for a in first_agents if a != 'p']
    
    for t in range(T - 1):
        if t + 1 >= len(states):
            break
        
        # 跳过第1年末
        if skip_year1 and t == 11:
            skipped_year1 += len(agent_ids)
            continue
        
        interest_rate = get_interest_rate(t, world)
        is_year_start = interest_rate is not None
        
        if is_year_start:
            interest_periods += 1
        
        for agent_id in agent_ids:
            if agent_id not in states[t] or agent_id not in states[t+1]:
                continue
            if agent_id not in periodic_tax[t]:
                continue
            
            total_checks += 1
            
            W_t = float(states[t][agent_id]['inventory'].get('Coin', 0.0))
            W_t1 = float(states[t+1][agent_id]['inventory'].get('Coin', 0.0))
            
            income_t = float(periodic_tax[t][agent_id].get('income', 0.0))
            tax_t = float(periodic_tax[t][agent_id].get('tax_paid', 0.0))
            lump_t = float(periodic_tax[t][agent_id].get('lump_sum', 0.0))
            consumption_t = float(states[t+1][agent_id]['consumption'].get('Coin', 0.0))
            
            lhs = W_t1 - W_t
            rhs = income_t - tax_t + lump_t - consumption_t
            
            if is_year_start:
                interest = W_t * interest_rate
                rhs += interest
            else:
                interest = 0.0
            
            error = abs(lhs - rhs)
            all_errors.append(error)
            
            if error > tolerance:
                violations.append({
                    't': t,
                    'agent': agent_id,
                    'year': t // 12 + 1,
                    'month': t % 12 + 1,
                    'W_t': W_t,
                    'W_t+1': W_t1,
                    'income': income_t,
                    'tax': tax_t,
                    'lump_sum': lump_t,
                    'consumption': consumption_t,
                    'interest': interest if is_year_start else None,
                    'interest_rate': float(interest_rate) if is_year_start else None,
                    'is_year_start': is_year_start,
                    'lhs': lhs,
                    'rhs': rhs,
                    'error': error
                })
    
    summary = {
        'total_checks': total_checks,
        'skipped_year1': skipped_year1,
        'interest_periods': interest_periods,
        'violations_count': len(violations),
        'violation_rate': len(violations) / total_checks if total_checks > 0 else 0,
        'max_error': max(all_errors) if all_errors else 0.0,
        'mean_error': statistics.mean(all_errors) if all_errors else 0.0,
        'median_error': statistics.median(all_errors) if all_errors else 0.0,
        'mean_error_violations': statistics.mean([v['error'] for v in violations]) if violations else 0.0
    }
    
    return violations, summary


def print_report(violations: List[Dict], summary: Dict, top_k: int = 10):
    """打印测试报告"""
    print("=" * 80)
    print("EconAgent 预算约束测试报告（最终版 v3 - 排除第1年）")
    print("=" * 80)
    
    print(f"\n总检查次数: {summary['total_checks']:,}")
    print(f"跳过的检查 (第1年末): {summary['skipped_year1']:,}")
    print(f"年初结算次数: {summary['interest_periods']:,}")
    print(f"违反次数: {summary['violations_count']:,}")
    print(f"违反率: {summary['violation_rate']:.6%}")
    print(f"最大误差: ${summary['max_error']:.2f}")
    print(f"平均误差 (所有): ${summary['mean_error']:.6f}")
    print(f"中位数误差 (所有): ${summary['median_error']:.6f}")
    
    if summary['violations_count'] == 0:
        print("\n✅ 测试通过！所有预算约束均满足！")
    else:
        print(f"\n⚠️  发现 {summary['violations_count']} 个违反情况")
        print(f"平均误差 (仅违反): ${summary['mean_error_violations']:.2f}")
        
        print(f"\n前 {min(top_k, len(violations))} 个最严重的违反情况:")
        print("-" * 80)
        
        sorted_violations = sorted(violations, key=lambda x: x['error'], reverse=True)[:top_k]
        
        for rank, v in enumerate(sorted_violations, 1):
            print(f"\n第 {rank} 名:")
            print(f"  时间: 第{v['year']}年第{v['month']}月 (t={v['t']}), Agent: {v['agent']}")
            if v.get('is_year_start'):
                print(f"  [年初结算，利率={v['interest_rate']:.6f}]")
            print(f"  ΔW = ${v['lhs']:,.2f}")
            print(f"  预期 = ${v['rhs']:,.2f}")
            print(f"  误差 = ${v['error']:,.2f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="EconAgent 预算约束测试（排除第1年）"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pickle", type=str, default="dense_log.pkl")
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--save_csv", type=str, default=None)
    
    args = parser.parse_args()
    
    pkl_path = os.path.join(args.data_path, args.pickle)
    print("正在加载数据...")
    with open(pkl_path, 'rb') as f:
        dense_log = pkl.load(f)
    print("数据加载完成！\n")
    
    violations, summary = test_budget_constraint_final(
        dense_log, 
        tolerance=args.tolerance,
        skip_year1=True
    )
    
    print_report(violations, summary, top_k=args.top_k)
    
    if args.save_json:
        results = {'summary': summary, 'violations': violations}
        json_path = os.path.join(args.data_path, args.save_json)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {json_path}")
    
    if args.save_csv and len(violations) > 0:
        csv_path = os.path.join(args.data_path, args.save_csv)
        fieldnames = ['t', 'year', 'month', 'agent', 'error', 'is_year_start', 'interest_rate']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for v in violations:
                writer.writerow({k: v.get(k, '') for k in fieldnames})
        print(f"✅ 违反情况已保存到: {csv_path}")


if __name__ == "__main__":
    main()