# -*- coding: utf-8 -*-
"""
consumption_inequality_convexity_test_FIXED.py

测试 EconAgent 的消费不平等凸性约束（修复重复计数问题）
"""

import os
import argparse
import pickle as pkl
import json
import csv
import statistics
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
# import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict


def extract_consumption_by_age(
    dense_log: Dict[str, Any],
    method: str = 'average'
) -> Dict[int, List[float]]:
    """
    按年龄提取消费数据（修复重复计数问题）
    
    Args:
        method: 'average' - 取年度平均（推荐）
                'first' - 只取第一次记录
                'random' - 随机抽取一个月
    
    Returns:
        age_consumption: {age: [consumption_list]}
    """
    states = dense_log.get('states', [])
    
    if not states:
        raise ValueError("states 数据不存在")
    
    # 收集每个 (agent, age) 的所有消费
    agent_age_consumptions = defaultdict(list)
    
    for t in range(len(states)):
        for aid in states[t].keys():
            if not aid.isdigit():
                continue
            
            agent_state = states[t][aid]
            
            # 提取年龄
            age = agent_state['endogenous'].get('age', None)
            
            # 提取消费
            consumption = agent_state['consumption'].get('Coin', None)
            
            if age is None or consumption is None:
                continue
            
            # 过滤掉0消费（可能是数据异常）
            if consumption <= 0:
                continue
            
            key = (aid, int(age))
            agent_age_consumptions[key].append(float(consumption))
    
    # 按方法处理，确保每个 (agent, age) 只有一个值
    age_consumption = {}
    
    for (aid, age), consumptions in agent_age_consumptions.items():
        if method == 'average':
            value = np.mean(consumptions)
        elif method == 'first':
            value = consumptions[0]
        elif method == 'random':
            value = np.random.choice(consumptions)
        else:
            raise ValueError(f"未知方法: {method}")
        
        if age not in age_consumption:
            age_consumption[age] = []
        
        age_consumption[age].append(value)
    
    return age_consumption


def compute_age_inequality_profile(
    age_consumption: Dict[int, List[float]],
    min_age: int = 18,
    max_age: int = 60
) -> Tuple[List[int], List[float], List[int]]:
    """
    计算年龄-不平等曲线
    
    V(t) = log(Var(log(C_i^t)))
    
    Returns:
        ages: 年龄列表
        V_values: 不平等指标值
        sample_sizes: 每个年龄的样本数
    """
    ages = []
    V_values = []
    sample_sizes = []
    
    for age in range(min_age, max_age + 1):
        if age not in age_consumption:
            continue
        
        consumptions = age_consumption[age]
        
        # 至少需要3个样本才能计算方差
        if len(consumptions) < 3:
            continue
        
        # 计算 log(consumption)
        log_consumptions = [np.log(c) for c in consumptions]
        
        # 计算方差
        variance = np.var(log_consumptions, ddof=1)  # 使用样本方差
        
        # 如果方差为0或负数，跳过
        if variance <= 0:
            continue
        
        # V(t) = log(Var(log(C)))
        V = np.log(variance)
        
        ages.append(age)
        V_values.append(V)
        sample_sizes.append(len(consumptions))
    
    return ages, V_values, sample_sizes


def fit_quadratic_polynomial(
    ages: List[int],
    V_values: List[float],
    base_age: int = 18
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    拟合二次多项式：V(t) = a0 + a1*(t-base) + a2*(t-base)^2
    
    Returns:
        coefficients: [a0, a1, a2]
        confidence_intervals: 95% 置信区间
        fit_stats: 拟合统计信息
    """
    if len(ages) < 3:
        raise ValueError(f"样本数不足：至少需要3个年龄组，当前只有{len(ages)}个")
    
    # 标准化年龄（减去基准年龄）
    ages_normalized = np.array(ages) - base_age
    V_array = np.array(V_values)
    
    # 构建设计矩阵 [1, (t-base), (t-base)^2]
    X = np.column_stack([
        np.ones(len(ages_normalized)),
        ages_normalized,
        ages_normalized ** 2
    ])
    
    # 最小二乘拟合
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        X, V_array, rcond=None
    )
    
    # 计算残差标准误
    n = len(ages)
    p = 3  # 参数个数
    
    if n > p:
        residual_std_error = np.sqrt(np.sum((V_array - X @ coefficients)**2) / (n - p))
    else:
        residual_std_error = 0
    
    # 计算标准误和置信区间
    # SE(β) = sqrt(σ² * (X'X)^(-1))
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        standard_errors = residual_std_error * np.sqrt(np.diag(XtX_inv))
        
        # 95% 置信区间 (使用 t 分布)
        t_critical = stats.t.ppf(0.975, n - p)
        confidence_intervals = np.array([
            [coef - t_critical * se, coef + t_critical * se]
            for coef, se in zip(coefficients, standard_errors)
        ])
    except np.linalg.LinAlgError:
        # 如果矩阵不可逆，使用 bootstrap
        print("警告：无法计算解析置信区间，将使用 bootstrap")
        confidence_intervals = bootstrap_confidence_intervals(
            ages, V_values, base_age, n_bootstrap=500
        )
        t_critical = stats.t.ppf(0.975, n - p)
        standard_errors = (confidence_intervals[:, 1] - confidence_intervals[:, 0]) / (2 * t_critical)
    
    # 计算 R²
    SS_res = np.sum((V_array - X @ coefficients)**2)
    SS_tot = np.sum((V_array - np.mean(V_array))**2)
    R_squared = 1 - (SS_res / SS_tot) if SS_tot > 0 else 0
    
    # 拟合统计信息
    fit_stats = {
        'n_obs': n,
        'R_squared': R_squared,
        'residual_std_error': residual_std_error,
        'standard_errors': standard_errors.tolist()
    }
    
    return coefficients, confidence_intervals, fit_stats


def bootstrap_confidence_intervals(
    ages: List[int],
    V_values: List[float],
    base_age: int = 18,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95
) -> np.ndarray:
    """
    使用 Bootstrap 计算置信区间
    
    Returns:
        confidence_intervals: shape (3, 2) for [a0, a1, a2]
    """
    n = len(ages)
    bootstrap_coefficients = []
    
    ages_array = np.array(ages)
    V_array = np.array(V_values)
    
    for _ in range(n_bootstrap):
        # 重采样
        indices = np.random.choice(n, size=n, replace=True)
        ages_boot = ages_array[indices] - base_age
        V_boot = V_array[indices]
        
        # 拟合
        X_boot = np.column_stack([
            np.ones(n),
            ages_boot,
            ages_boot ** 2
        ])
        
        try:
            coef_boot, _, _, _ = np.linalg.lstsq(X_boot, V_boot, rcond=None)
            bootstrap_coefficients.append(coef_boot)
        except:
            continue
    
    bootstrap_coefficients = np.array(bootstrap_coefficients)
    
    # 计算百分位数
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    confidence_intervals = np.array([
        [
            np.percentile(bootstrap_coefficients[:, i], lower_percentile),
            np.percentile(bootstrap_coefficients[:, i], upper_percentile)
        ]
        for i in range(3)
    ])
    
    return confidence_intervals


def test_convexity_constraint(
    a2: float,
    ci_lower: float,
    ci_upper: float,
    lower_bound: float = 2.6e-4,
    upper_bound: float = 3.2e-3
) -> Dict[str, Any]:
    """
    测试凸性约束
    
    Returns:
        result: 测试结果字典
    """
    # 点估计是否在范围内
    point_satisfies = (lower_bound <= a2 <= upper_bound)
    
    # 置信区间是否完全在范围内
    ci_satisfies = (lower_bound <= ci_lower) and (ci_upper <= upper_bound)
    
    # 置信区间是否与约束范围有交集
    ci_overlaps = (ci_lower <= upper_bound) and (ci_upper >= lower_bound)
    
    # 判断违反类型
    if a2 < 0:
        violation_type = 'severe'
        violation_desc = '严重违反（凹性曲线）'
    elif 0 <= a2 < lower_bound:
        violation_type = 'weak'
        violation_desc = '轻微违反（过于平坦）'
    elif a2 > upper_bound:
        violation_type = 'excessive'
        violation_desc = '过度凸性（增长过快）'
    else:
        violation_type = None
        violation_desc = '满足约束'
    
    result = {
        'a2': a2,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'constraint_lower': lower_bound,
        'constraint_upper': upper_bound,
        'point_satisfies': point_satisfies,
        'ci_satisfies': ci_satisfies,
        'ci_overlaps': ci_overlaps,
        'violation_type': violation_type,
        'violation_desc': violation_desc
    }
    
    return result


def print_convexity_report(
    ages: List[int],
    V_values: List[float],
    sample_sizes: List[int],
    coefficients: np.ndarray,
    confidence_intervals: np.ndarray,
    fit_stats: Dict[str, Any],
    test_result: Dict[str, Any],
    method: str
):
    """打印消费不平等凸性测试报告"""
    print("=" * 80)
    print("消费不平等凸性约束测试报告（修复版）")
    print("=" * 80)
    
    print(f"\n数据处理方法：{method}")
    if method == 'average':
        print(f"  每个agent每个年龄取年度平均消费")
    elif method == 'first':
        print(f"  每个agent每个年龄只取第一次记录")
    elif method == 'random':
        print(f"  每个agent每个年龄随机抽取一个月")
    
    print(f"\n理论背景：")
    print(f"  数据来源: Guvenen (2007)")
    print(f"  理论基础: 收入冲击和个体收入曲线学习的累积效应")
    print(f"  预期模式: 消费不平等随年龄呈凸性（加速）增长")
    
    print(f"\n数学约束：")
    print(f"  V(t) = a₀ + a₁·(t-18) + a₂·(t-18)²")
    print(f"  要求: {test_result['constraint_lower']:.2e} ≤ a₂ ≤ {test_result['constraint_upper']:.2e}")
    
    print(f"\n数据统计：")
    print(f"  年龄范围: {min(ages)} - {max(ages)} 岁")
    print(f"  年龄组数: {len(ages)}")
    print(f"  总样本数: {sum(sample_sizes)}")
    print(f"  平均每组样本数: {np.mean(sample_sizes):.1f}")
    
    print(f"\n拟合结果：")
    print(f"  R²: {fit_stats['R_squared']:.4f}")
    print(f"  残差标准误: {fit_stats['residual_std_error']:.4f}")
    
    print(f"\n回归系数（95% 置信区间）：")
    a0, a1, a2 = coefficients
    ci_a0, ci_a1, ci_a2 = confidence_intervals
    
    print(f"  a₀ (截距): {a0:.4f} [{ci_a0[0]:.4f}, {ci_a0[1]:.4f}]")
    print(f"  a₁ (线性项): {a1:.4f} [{ci_a1[0]:.4f}, {ci_a1[1]:.4f}]")
    print(f"  a₂ (二次项): {a2:.6f} [{ci_a2[0]:.6f}, {ci_a2[1]:.6f}]")
    print(f"             (科学计数法: {a2:.2e} [{ci_a2[0]:.2e}, {ci_a2[1]:.2e}])")
    
    print(f"\n约束测试结果：")
    print(f"  点估计满足: {'是' if test_result['point_satisfies'] else '否'}")
    print(f"  置信区间满足: {'是' if test_result['ci_satisfies'] else '否'}")
    print(f"  置信区间与约束重叠: {'是' if test_result['ci_overlaps'] else '否'}")
    
    # 判断是否通过
    if test_result['point_satisfies']:
        print(f"\n✅ 测试通过！")
        print(f"   二次系数 a₂ = {a2:.2e} 在约束范围内")
        print(f"   消费不平等随年龄呈适度凸性增长")
    else:
        print(f"\n❌ 测试失败！")
        print(f"   {test_result['violation_desc']}")
        print(f"   a₂ = {a2:.2e}")
        
        if test_result['violation_type'] == 'severe':
            print(f"   → 消费不平等随年龄呈凹性增长（与理论相反）")
        elif test_result['violation_type'] == 'weak':
            print(f"   → 消费不平等增长过于平缓")
            print(f"   → 与实证数据的凸性模式不符")
        elif test_result['violation_type'] == 'excessive':
            print(f"   → 消费不平等增长过快")
            print(f"   → 超出实证数据观察范围")
    
    # 样本量警告
    if min(sample_sizes) < 30:
        print(f"\n⚠️  警告：部分年龄组样本量较小（最小 {min(sample_sizes)}）")
        print(f"   建议：增加模拟时长或agent数量以提高统计可靠性")
    
    print("\n" + "=" * 80)


def plot_age_inequality_profile(
    ages: List[int],
    V_values: List[float],
    sample_sizes: List[int],
    coefficients: np.ndarray,
    save_path: Optional[str] = None
):
    """绘制年龄-不平等曲线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：年龄-不平等曲线 + 拟合曲线
    ax1.scatter(ages, V_values, s=50, alpha=0.6, label='实际数据')
    
    # 绘制拟合曲线
    ages_fit = np.linspace(min(ages), max(ages), 100)
    ages_fit_norm = ages_fit - 18
    V_fit = coefficients[0] + coefficients[1] * ages_fit_norm + coefficients[2] * ages_fit_norm**2
    
    ax1.plot(ages_fit, V_fit, 'r-', linewidth=2, label='二次拟合')
    
    ax1.set_xlabel('年龄', fontsize=12)
    ax1.set_ylabel('V(t) = log(Var(log(C)))', fontsize=12)
    ax1.set_title('年龄-消费不平等曲线（修复版）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：每个年龄的样本量
    ax2.bar(ages, sample_sizes, alpha=0.6, color='steelblue')
    ax2.set_xlabel('年龄', fontsize=12)
    ax2.set_ylabel('样本数', fontsize=12)
    ax2.set_title('各年龄组样本分布', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
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
        description="EconAgent 消费不平等凸性约束测试（修复版）"
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
        "--min_age",
        type=int,
        default=18,
        help="最小年龄"
    )
    parser.add_argument(
        "--max_age",
        type=int,
        default=60,
        help="最大年龄"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="average",
        choices=['average', 'first', 'random'],
        help="处理重复数据的方法"
    )
    parser.add_argument(
        "--lower_bound",
        type=float,
        default=2.6e-4,
        help="a2 下界"
    )
    parser.add_argument(
        "--upper_bound",
        type=float,
        default=3.2e-3,
        help="a2 上界"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Bootstrap 重采样次数（0表示不使用）"
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
        help="保存年龄数据到CSV"
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
    
    # 提取数据（使用修复后的方法）
    print(f"提取消费数据（方法: {args.method}）...")
    age_consumption = extract_consumption_by_age(dense_log, method=args.method)
    print(f"已提取 {len(age_consumption)} 个年龄组的数据\n")
    
    # 计算年龄-不平等曲线
    print("计算年龄-不平等曲线...")
    ages, V_values, sample_sizes = compute_age_inequality_profile(
        age_consumption,
        min_age=args.min_age,
        max_age=args.max_age
    )
    print(f"有效年龄组: {len(ages)}\n")
    
    if len(ages) < 3:
        raise ValueError(f"年龄组数量不足：至少需要3个，当前只有{len(ages)}个")
    
    # 拟合二次多项式
    print("拟合二次多项式...")
    coefficients, confidence_intervals, fit_stats = fit_quadratic_polynomial(
        ages, V_values, base_age=args.min_age
    )
    print("拟合完成！\n")
    
    # 测试约束
    a2 = coefficients[2]
    ci_a2 = confidence_intervals[2]
    
    test_result = test_convexity_constraint(
        a2, ci_a2[0], ci_a2[1],
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound
    )
    
    # 打印报告
    print_convexity_report(
        ages, V_values, sample_sizes,
        coefficients, confidence_intervals, fit_stats,
        test_result, method=args.method
    )
    
    # 绘制图表
    if args.plot:
        plot_path = os.path.join(args.data_path, args.plot)
        plot_age_inequality_profile(
            ages, V_values, sample_sizes, coefficients,
            save_path=plot_path
        )
    
    # 保存结果
    if args.save_json:
        results = {
            'method': args.method,
            'coefficients': {
                'a0': float(coefficients[0]),
                'a1': float(coefficients[1]),
                'a2': float(coefficients[2])
            },
            'confidence_intervals': {
                'a0': [float(confidence_intervals[0, 0]), float(confidence_intervals[0, 1])],
                'a1': [float(confidence_intervals[1, 0]), float(confidence_intervals[1, 1])],
                'a2': [float(confidence_intervals[2, 0]), float(confidence_intervals[2, 1])]
            },
            'fit_stats': fit_stats,
            'test_result': test_result,
            'age_data': {
                'ages': ages,
                'V_values': V_values,
                'sample_sizes': sample_sizes
            }
        }
        json_path = os.path.join(args.data_path, args.save_json)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {json_path}")
    
    if args.save_csv:
        csv_path = os.path.join(args.data_path, args.save_csv)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['age', 'V_value', 'sample_size'])
            for age, v, n in zip(ages, V_values, sample_sizes):
                writer.writerow([age, v, n])
        print(f"✅ 年龄数据已保存到: {csv_path}")


if __name__ == "__main__":
    main()