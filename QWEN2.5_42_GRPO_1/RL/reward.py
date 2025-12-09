
import json
import re

def interval_score(x: float, low: float, high: float, margin: float = 0.1) -> float:
    if low <= x <= high:
        return 1.0
    if x < low:
        if x <= low - margin:
            return 0.0
        return (x - (low - margin)) / margin
    else:
        if x >= high + margin:
            return 0.0
        return ((high + margin) - x) / margin


def parse_action(response: str):
    try:
        text = response.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except:
        return None


_DEBUG_COUNT = 0

def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float:
    """
    verl reward函数
    """
    global _DEBUG_COUNT
    _DEBUG_COUNT += 1
    
    # 调试：只打印前5个
    if _DEBUG_COUNT <= 5:
        with open('/workspace/QWEN2.5_42_GRPO_1/reward_debug.log', 'a') as f:
            f.write(f"\n=== DEBUG {_DEBUG_COUNT} ===\n")
            f.write(f"data_source: {data_source}\n")
            f.write(f"solution_str type: {type(solution_str)}\n")
            f.write(f"solution_str: {repr(solution_str)[:500]}\n")
            f.write(f"extra_info: {repr(extra_info)[:200]}\n")
    
    if data_source != "econ_agent":
        return 0.0
    
    reward = 0.0
    
    action = parse_action(solution_str)
    if action is None:
        if _DEBUG_COUNT <= 5:
            with open('/workspace/QWEN2.5_42_GRPO_1/reward_debug.log', 'a') as f:
                f.write(f"parse_action returned None!\n")
        return -1.0
    
    work = action.get("work")
    consumption = action.get("consumption")
    
    if work is None or consumption is None:
        return -0.8
    
    try:
        work = float(work)
        consumption = float(consumption)
    except:
        return -0.8
    
    reward += 0.1
    
    if 0 <= work <= 1:
        reward += 0.1
    else:
        reward -= 0.3
    
    if 0 <= consumption <= 1:
        reward += 0.1
    else:
        reward -= 0.3
    
    # Buffer-Stock 行为
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    
    income = extra_info.get('income', 1000)
    wealth = extra_info.get('wealth', 5000)
    dpi = extra_info.get('dpi', income)

    if 'buffer_ratio' in extra_info:
        buffer_ratio = extra_info['buffer_ratio']
    else:
        cash_on_hand = wealth + dpi
        buffer_ratio = cash_on_hand / (dpi + 1e-8) if dpi > 1e-6 else 1.0
    
    if buffer_ratio > 0:
        if buffer_ratio < 2:
            ideal_consumption = 0.3 + buffer_ratio * 0.1
        elif buffer_ratio <= 6:
            ideal_consumption = 0.5 + (buffer_ratio - 2) * 0.05
        else:
            ideal_consumption = min(0.7 + (buffer_ratio - 6) * 0.02, 0.85)
        
        consumption_diff = abs(consumption - ideal_consumption)
        if consumption_diff < 0.1:
            reward += 0.25
        elif consumption_diff < 0.2:
            reward += 0.15
        elif consumption_diff < 0.3:
            reward += 0.05
        else:
            reward -= 0.1
    
    saving_rate = 1 - consumption
    reward += interval_score(saving_rate, 0.014, 0.318, margin=0.05) * 0.15
    
    if 0.2 <= consumption <= 0.8:
        reward += 0.1
    elif 0.1 <= consumption <= 0.9:
        reward += 0.05
    else:
        reward -= 0.1
    
    if income > 0 or extra_info.get('skill', 0) > 0:
        if work >= 0.5:
            reward += 0.1
    
    if buffer_ratio < 2 and work >= 0.7:
        reward += 0.1
    
    return max(-1.0, min(1.0, reward))
