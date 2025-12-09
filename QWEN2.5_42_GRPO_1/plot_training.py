#!/usr/bin/env python3
"""
从 training.log 提取训练指标并生成图表
"""

import re
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_path):
    """解析训练日志，提取关键指标"""
    
    steps = []
    scores = []
    pg_losses = []
    grad_norms = []
    entropies = []
    times_per_step = []
    
    # 正则表达式匹配 step 行
    step_pattern = re.compile(
        r'step:(\d+).*?'
        r'actor/entropy:([\d.]+).*?'
        r'actor/pg_loss:np\.float64\(([-\d.]+)\).*?'
        r'actor/grad_norm:np\.float64\(([\d.]+)\).*?'
        r'critic/score/mean:([\d.]+).*?'
        r'perf/time_per_step:([\d.]+)'
    )
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # 查找所有匹配
    for match in step_pattern.finditer(content):
        step = int(match.group(1))
        entropy = float(match.group(2))
        pg_loss = float(match.group(3))
        grad_norm = float(match.group(4))
        score = float(match.group(5))
        time_per_step = float(match.group(6))
        
        steps.append(step)
        entropies.append(entropy)
        pg_losses.append(pg_loss)
        grad_norms.append(grad_norm)
        scores.append(score)
        times_per_step.append(time_per_step)
    
    return {
        'steps': np.array(steps),
        'scores': np.array(scores),
        'pg_losses': np.array(pg_losses),
        'grad_norms': np.array(grad_norms),
        'entropies': np.array(entropies),
        'times_per_step': np.array(times_per_step)
    }

def smooth(data, window=5):
    """简单移动平均平滑"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_metrics(data, output_dir='.'):
    """生成训练曲线图"""
    
    if len(data['steps']) == 0:
        print("没有找到训练数据！")
        return
    
    print(f"找到 {len(data['steps'])} 步训练数据")
    print(f"Step 范围: {data['steps'].min()} - {data['steps'].max()}")
    
    # 设置中文字体（如果可用）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GRPO Training Metrics (Qwen2.5-7B Economic Agent)', fontsize=14, fontweight='bold')
    
    # 1. Reward (Score) 曲线
    ax1 = axes[0, 0]
    ax1.plot(data['steps'], data['scores'], 'b-', alpha=0.3, label='Raw')
    if len(data['scores']) >= 5:
        smooth_scores = smooth(data['scores'], 5)
        smooth_steps = data['steps'][2:-2] if len(data['steps']) > 4 else data['steps'][:len(smooth_scores)]
        ax1.plot(smooth_steps, smooth_scores, 'b-', linewidth=2, label='Smoothed (window=5)')
    ax1.axhline(y=data['scores'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {data["scores"].mean():.3f}')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Reward (Score)')
    ax1.set_title('Reward Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. Entropy 曲线
    ax2 = axes[0, 1]
    ax2.plot(data['steps'], data['entropies'], 'g-', alpha=0.3, label='Raw')
    if len(data['entropies']) >= 5:
        smooth_entropy = smooth(data['entropies'], 5)
        smooth_steps = data['steps'][2:-2] if len(data['steps']) > 4 else data['steps'][:len(smooth_entropy)]
        ax2.plot(smooth_steps, smooth_entropy, 'g-', linewidth=2, label='Smoothed (window=5)')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Policy Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Policy Gradient Loss 曲线
    ax3 = axes[1, 0]
    ax3.plot(data['steps'], data['pg_losses'], 'r-', alpha=0.3, label='Raw')
    if len(data['pg_losses']) >= 5:
        smooth_loss = smooth(data['pg_losses'], 5)
        smooth_steps = data['steps'][2:-2] if len(data['steps']) > 4 else data['steps'][:len(smooth_loss)]
        ax3.plot(smooth_steps, smooth_loss, 'r-', linewidth=2, label='Smoothed (window=5)')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('PG Loss')
    ax3.set_title('Policy Gradient Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Gradient Norm 曲线
    ax4 = axes[1, 1]
    ax4.plot(data['steps'], data['grad_norms'], 'm-', alpha=0.3, label='Raw')
    if len(data['grad_norms']) >= 5:
        smooth_grad = smooth(data['grad_norms'], 5)
        smooth_steps = data['steps'][2:-2] if len(data['steps']) > 4 else data['steps'][:len(smooth_grad)]
        ax4.plot(smooth_steps, smooth_grad, 'm-', linewidth=2, label='Smoothed (window=5)')
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Grad Clip (1.0)')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Gradient Norm')
    ax4.set_title('Gradient Norm')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = f'{output_dir}/training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 打印统计信息
    print("\n=== 训练统计 ===")
    print(f"总步数: {len(data['steps'])}")
    print(f"Reward - Mean: {data['scores'].mean():.4f}, Min: {data['scores'].min():.4f}, Max: {data['scores'].max():.4f}")
    print(f"Entropy - Mean: {data['entropies'].mean():.4f}, Min: {data['entropies'].min():.4f}, Max: {data['entropies'].max():.4f}")
    print(f"PG Loss - Mean: {data['pg_losses'].mean():.4f}")
    print(f"Grad Norm - Mean: {data['grad_norms'].mean():.4f}, Max: {data['grad_norms'].max():.4f}")
    print(f"Time/Step - Mean: {data['times_per_step'].mean():.2f}s")
    
    plt.close()
    
    return output_path

if __name__ == '__main__':
    import sys
    
    log_path = '/workspace/training.log'
    output_dir = '/workspace/QWEN2.5_42_GRPO_1'
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    
    print(f"解析日志: {log_path}")
    data = parse_training_log(log_path)
    
    if len(data['steps']) > 0:
        plot_metrics(data, output_dir)
    else:
        print("未找到训练数据，请检查日志文件格式")
