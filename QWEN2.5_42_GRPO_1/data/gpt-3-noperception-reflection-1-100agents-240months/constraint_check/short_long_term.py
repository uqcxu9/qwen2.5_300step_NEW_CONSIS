# -*- coding: utf-8 -*-
"""
income_wealth_preference_test.py

测试 EconAgent 的短期收入 vs 长期财富偏好

根据心理账户理论，人们在做劳动决策时更关注即时收入而非长期财富积累。

约束条件：
1. 近期收入对劳动决策的影响 > 当前财富的影响
2. 数学公式：Labor = σ(a + b1*Wealth + b2*RecentIncome)
3. 假设：|b2| > |b1| 且 b2 显著 (p < 0.05)

近期收入计算（EMA）：
    π_recent[t] = λ * income[t] + (1-λ) * π_recent[t-1]
"""

import os
import argparse
import pickle as pkl
import json
import csv
import statistics
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats


def compute_recent_income_ema(
    incomes: List[float],
    lambda_param: float = 0.3
) -> List[float]:
    """
    计算近期收入的指数移动平均 (EMA)
    
    π_recent[t] = λ * income[t] + (1-λ) * π_recent[t-1]
    π_recent[0] = income[0]
    
    Args:
        incomes: 收入时间序列
        lambda_param: EMA 平滑参数 (0-1)
    
    Returns:
        近期收入序列
    """
    if not incomes:
        return []
    
    recent_incomes = [incomes[0]]  # 初始化
    
    for t in range(1, len(incomes)):
        recent = lambda_param * incomes[t] + (1 - lambda_param) * recent_incomes[t-1]
        recent_incomes.append(recent)
    
    return recent_incomes


def prepare_agent_data(
    dense_log: Dict[str, Any],
    agent_id: str,
    lambda_param: float = 0.3
) -> Tuple[List[float], List[float], List[int]]:
    """
    为单个agent准备回归数据
    
    Returns:
        wealth_series: 财富时间序列
        recent_income_series: 近期收入时间序列
        labor_series: 劳动决策时间序列 (0/1)
    """
    states = dense_log['states']
    
    wealth_series = []
    income_series = []
    labor_series = []
    
    T = len(states)
    
    for t in range(T):
        # 检查数据是否存在
        if agent_id not in states[t]:
            continue
        
        # 提取财富
        wealth = float(states[t][agent_id]['inventory'].get('Coin', 0.0))
        
        # 提取收入
        income = float(states[t][agent_id]['income'].get('Coin', 0.0))
        
        # 提取劳动决策（从 endogenous）
        labor_value = states[t][agent_id]['endogenous'].get('Labor', None)
        
        if labor_value is None:
            continue
        
        # 将劳动决策转换为二元变量 (0/1)
        # Labor > 0 表示工作
        labor_binary = 1 if labor_value > 0 else 0
        
        wealth_series.append(wealth)
        income_series.append(income)
        labor_series.append(labor_binary)
    
    # 计算近期收入 (EMA)
    recent_income_series = compute_recent_income_ema(income_series, lambda_param)
    
    return wealth_series, recent_income_series, labor_series


def run_logistic_regression(
    wealth: List[float],
    recent_income: List[float],
    labor: List[int]
) -> Optional[Dict[str, Any]]:
    """
    运行 logistic 回归：Labor ~ Wealth + RecentIncome
    
    Returns:
        回归结果字典，包含系数、p值等
    """
    if len(wealth) < 10:  # 至少需要10个数据点
        return None
    
    # 标准化特征（重要！）
    wealth_array = np.array(wealth).reshape(-1, 1)
    recent_income_array = np.array(recent_income).reshape(-1, 1)
    labor_array = np.array(labor)
    
    # 标准化
    wealth_mean, wealth_std = np.mean(wealth_array), np.std(wealth_array)
    income_mean, income_std = np.mean(recent_income_array), np.std(recent_income_array)
    
    if wealth_std == 0 or income_std == 0:
        return None
    
    wealth_normalized = (wealth_array - wealth_mean) / wealth_std
    income_normalized = (recent_income_array - income_mean) / income_std
    
    # 构建特征矩阵
    X = np.hstack([wealth_normalized, income_normalized])
    y = labor_array
    
    # 检查是否有变化（如果所有y都相同，无法回归）
    if len(np.unique(y)) < 2:
        return None
    
    # Logistic 回归
    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # 提取系数
        b1_wealth = float(model.coef_[0, 0])
        b2_income = float(model.coef_[0, 1])
        intercept = float(model.intercept_[0])
        
        # 计算 p 值（使用 z-test 近似）
        # 注意：sklearn 的 LogisticRegression 不直接提供 p 值
        # 这里使用 statsmodels 会更准确，但为了简化使用近似方法
        
        # 预测准确率
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)
        
        # 使用 Wald test 近似 p 值
        # p_value ≈ 2 * (1 - Φ(|z|)), where z = coef / se
        # 简化版本：使用准确率作为显著性指标
        
        return {
            'b1_wealth': b1_wealth,
            'b2_recent_income': b2_income,
            'intercept': intercept,
            'accuracy': accuracy,
            'n_samples': len(labor),
            'labor_rate': np.mean(labor),  # 工作比例
            'abs_b2_gt_abs_b1': bool(abs(b2_income) > abs(b1_wealth))  # ← 加 bool()
        }
    
    except Exception as e:
        print(f"回归失败: {e}")
        return None


def test_income_wealth_preference(
    dense_log: Dict[str, Any],
    lambda_param: float = 0.3,
    min_agents: int = 10
) -> Tuple[List[Dict], Dict]:
    """
    测试所有agent的收入-财富偏好
    
    Args:
        dense_log: 完整日志数据
        lambda_param: EMA 平滑参数
        min_agents: 最少需要的有效agent数量
    
    Returns:
        agent_results: 每个agent的回归结果
        summary: 汇总统计
    """
    states = dense_log.get('states', [])
    
    if not states:
        raise ValueError("states 数据不存在")
    
    # 获取所有agent
    agent_ids = [aid for aid in states[0].keys() if aid != 'p']
    
    agent_results = []
    
    print(f"正在分析 {len(agent_ids)} 个agent...")
    
    for i, agent_id in enumerate(agent_ids):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(agent_ids)}")
        
        # 准备数据
        wealth, recent_income, labor = prepare_agent_data(
            dense_log, agent_id, lambda_param
        )
        
        if len(labor) < 10:
            continue
        
        # 运行回归
        result = run_logistic_regression(wealth, recent_income, labor)
        
        if result is not None:
            result['agent_id'] = agent_id
            agent_results.append(result)
    
    if len(agent_results) < min_agents:
        raise ValueError(f"有效agent数量不足：{len(agent_results)} < {min_agents}")
    
    # 汇总统计
    b1_values = [r['b1_wealth'] for r in agent_results]
    b2_values = [r['b2_recent_income'] for r in agent_results]
    abs_b1_values = [abs(b1) for b1 in b1_values]
    abs_b2_values = [abs(b2) for b2 in b2_values]
    
    # Paired t-test: |b2| vs |b1|
    t_stat, p_value = stats.ttest_rel(abs_b2_values, abs_b1_values)
    
    # 计算有多少agent满足 |b2| > |b1|
    count_b2_greater = sum(1 for r in agent_results if r['abs_b2_gt_abs_b1'])
    
    summary = {
        'total_agents': len(agent_results),
        'lambda_param': lambda_param,
        
        # b1 (wealth) 统计
        'mean_b1': statistics.mean(b1_values),
        'median_b1': statistics.median(b1_values),
        'std_b1': statistics.stdev(b1_values) if len(b1_values) > 1 else 0,
        'mean_abs_b1': statistics.mean(abs_b1_values),
        
        # b2 (recent income) 统计
        'mean_b2': statistics.mean(b2_values),
        'median_b2': statistics.median(b2_values),
        'std_b2': statistics.stdev(b2_values) if len(b2_values) > 1 else 0,
        'mean_abs_b2': statistics.mean(abs_b2_values),
        
        # 假设检验
        'count_b2_greater_b1': count_b2_greater,
        'rate_b2_greater_b1': count_b2_greater / len(agent_results),
        'paired_t_statistic': float(t_stat),
        'paired_p_value': float(p_value),
        'hypothesis_supported': (p_value < 0.05) and (t_stat > 0),
        
        # 平均准确率
        'mean_accuracy': statistics.mean([r['accuracy'] for r in agent_results]),
        'mean_labor_rate': statistics.mean([r['labor_rate'] for r in agent_results])
    }
    
    return agent_results, summary


def print_preference_report(
    agent_results: List[Dict],
    summary: Dict,
    top_k: int = 10
):
    """打印测试报告"""
    print("=" * 80)
    print("短期收入 vs 长期财富偏好测试报告")
    print("=" * 80)
    
    print(f"\n理论假设：")
    print(f"  心理账户理论认为：人们更关注近期收入而非长期财富")
    print(f"  数学假设：|b2| > |b1|，其中 b1=财富系数, b2=近期收入系数")
    
    print(f"\n数据统计：")
    print(f"  分析agent数: {summary['total_agents']}")
    print(f"  EMA参数 λ: {summary['lambda_param']}")
    print(f"  平均劳动率: {summary['mean_labor_rate']:.2%}")
    print(f"  平均模型准确率: {summary['mean_accuracy']:.2%}")
    
    print(f"\n回归系数统计：")
    print(f"  b1 (财富系数):")
    print(f"    平均值: {summary['mean_b1']:.6f}")
    print(f"    中位数: {summary['median_b1']:.6f}")
    print(f"    标准差: {summary['std_b1']:.6f}")
    print(f"    平均|b1|: {summary['mean_abs_b1']:.6f}")
    
    print(f"\n  b2 (近期收入系数):")
    print(f"    平均值: {summary['mean_b2']:.6f}")
    print(f"    中位数: {summary['median_b2']:.6f}")
    print(f"    标准差: {summary['std_b2']:.6f}")
    print(f"    平均|b2|: {summary['mean_abs_b2']:.6f}")
    
    print(f"\n假设检验结果：")
    print(f"  |b2| > |b1| 的agent数: {summary['count_b2_greater_b1']} / {summary['total_agents']}")
    print(f"  |b2| > |b1| 的比例: {summary['rate_b2_greater_b1']:.2%}")
    
    print(f"\n  配对 t 检验 (|b2| vs |b1|):")
    print(f"    t 统计量: {summary['paired_t_statistic']:.4f}")
    print(f"    p 值: {summary['paired_p_value']:.6f}")
    
    if summary['hypothesis_supported']:
        print(f"\n✅ 假设得到支持！")
        print(f"   近期收入的影响显著大于长期财富 (p < 0.05)")
    else:
        if summary['paired_p_value'] >= 0.05:
            print(f"\n❌ 假设未得到支持")
            print(f"   差异不显著 (p = {summary['paired_p_value']:.4f} >= 0.05)")
        else:
            print(f"\n❌ 假设未得到支持")
            print(f"   |b1| > |b2| (与预期相反)")
    
    # 显示极端案例
    print(f"\n前 {min(top_k, len(agent_results))} 个近期收入影响最强的agent:")
    print("-" * 80)
    
    sorted_by_b2 = sorted(agent_results, key=lambda x: x['b2_recent_income'], reverse=True)[:top_k]
    
    for rank, r in enumerate(sorted_by_b2, 1):
        print(f"\n第 {rank} 名: Agent {r['agent_id']}")
        print(f"  b1 (财富): {r['b1_wealth']:.6f}")
        print(f"  b2 (近期收入): {r['b2_recent_income']:.6f}")
        print(f"  |b2| > |b1|: {'是' if r['abs_b2_gt_abs_b1'] else '否'}")
        print(f"  模型准确率: {r['accuracy']:.2%}")
        print(f"  样本数: {r['n_samples']}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="EconAgent 短期收入 vs 长期财富偏好测试"
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
        "--lambda_param",
        type=float,
        default=0.3,
        help="EMA平滑参数 (0-1)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="显示前K个agent"
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
        help="保存agent结果到CSV"
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
    
    # 运行测试
    print("开始分析...")
    agent_results, summary = test_income_wealth_preference(
        dense_log,
        lambda_param=args.lambda_param
    )
    
    # 打印报告
    print_preference_report(agent_results, summary, top_k=args.top_k)
    
    # 保存结果
    if args.save_json:
        results = {
            'summary': summary,
            'agent_results': agent_results
        }
        json_path = os.path.join(args.data_path, args.save_json)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {json_path}")
    
    if args.save_csv:
        csv_path = os.path.join(args.data_path, args.save_csv)
        fieldnames = ['agent_id', 'b1_wealth', 'b2_recent_income', 
                     'abs_b2_gt_abs_b1', 'accuracy', 'n_samples', 'labor_rate']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in agent_results:
                writer.writerow({k: r.get(k, '') for k in fieldnames})
        print(f"✅ Agent结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()