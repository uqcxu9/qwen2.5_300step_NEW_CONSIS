# prepare_verl_data.py
import pickle as pkl
import pandas as pd
import os
import json

# 经济决策 agent 的系统提示
SYSTEM_PROMPT = """You are an economic decision-making agent. Based on your current financial situation, you need to make two decisions:
1. work: A value between 0 and 1 representing your labor supply (0 = no work, 1 = full-time work)
2. consumption: A value between 0 and 1 representing the proportion of your disposable income to consume

You MUST respond with a valid JSON object in this exact format:
{"work": <float 0-1>, "consumption": <float 0-1>}

Do not include any other text or explanation, only output the JSON."""

def prepare_verl_dataset(data_dir, output_dir, num_agents=100):
    """
    从dense_log提取verl训练数据
    使用simulate时保存的真实prompt
    生成 verl 期望的 chat 格式数据
    """
    
    dense_log_path = f"{data_dir}/dense_log.pkl"
    with open(dense_log_path, 'rb') as f:
        dense_log = pkl.load(f)
    
    states = dense_log['states']
    actions = dense_log['actions']
    periodic_tax = dense_log['PeriodicTax']
    prompts = dense_log.get('prompts', [])  # 从dense_log读取prompt
    
    samples = []
    
    # 错误统计
    errors = {
        'missing_state': 0,
        'missing_prompt': 0,
        'missing_action': 0,
        'invalid_action_format': 0,
        'invalid_skill': 0,
    }
    
    for t in range(1, len(actions)):
        for agent_id in range(num_agents):
            agent_id_str = str(agent_id)
            
            # 检查 state 是否存在
            if agent_id_str not in states[t]:
                errors['missing_state'] += 1
                continue
            
            # 检查 prompt 是否存在 - 不返回默认值，记录错误
            if t >= len(prompts) or agent_id_str not in prompts[t]:
                errors['missing_prompt'] += 1
                continue
            
            raw_prompt = prompts[t][agent_id_str]
            if not raw_prompt or not isinstance(raw_prompt, str) or len(raw_prompt) < 100:
                errors['missing_prompt'] += 1
                continue
            
            state = states[t][agent_id_str]
            
            # 检查必要的 state 字段
            try:
                income = state['income']['Coin']
                wealth = state['inventory']['Coin']
                skill = state['skill']
            except (KeyError, TypeError):
                errors['missing_state'] += 1
                continue
            
            tax_info = periodic_tax[t].get(agent_id_str, {})
            tax_paid = tax_info.get('tax_paid', 0)
            lump_sum = tax_info.get('lump_sum', 0)
            
            dpi = income + lump_sum - tax_paid
            cash_on_hand = wealth + dpi
            buffer_ratio = cash_on_hand / (dpi + 1e-8) if dpi > 1e-6 else 1.0
            
            # 检查 action 是否存在 - 不返回默认值，记录错误
            action_data = actions[t].get(agent_id_str)
            if action_data is None:
                errors['missing_action'] += 1
                continue

            if isinstance(action_data, dict):
                work = action_data.get('SimpleLabor')
                cons_idx = action_data.get('SimpleConsumption')
                if work is None or cons_idx is None:
                    errors['invalid_action_format'] += 1
                    continue
            elif isinstance(action_data, (list, tuple)) and len(action_data) >= 2:
                work = action_data[0]
                cons_idx = action_data[1]
            else:
                errors['invalid_action_format'] += 1
                continue

            consumption_prop = cons_idx * 0.02
            
            # 构建 verl 期望的 chat 格式 prompt
            chat_prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt}
            ]
            
            extra_info_dict = {
                "timestep": t,
                "agent_id": agent_id,
                "income": float(income),
                "wealth": float(wealth),
                "dpi": float(dpi),
                "buffer_ratio": float(buffer_ratio),
                "tax_paid": float(tax_paid),
                "lump_sum": float(lump_sum),
                "skill": float(skill),
                "gt_work": int(work),
                "gt_consumption": float(consumption_prop),
            }
            
            samples.append({
                "prompt": chat_prompt,  # chat 格式的列表
                "data_source": "econ_agent",
                "extra_info": extra_info_dict,  # 字典格式（verl 要求）
                "reward_model": {"ground_truth": ""},  # verl 需要这个字段
                "ability": "economic_decision",  # 任务类型
            })
    
    # 打印错误统计
    print(f"\n=== 数据处理统计 ===")
    print(f"成功样本数: {len(samples)}")
    print(f"错误统计:")
    for err_type, count in errors.items():
        if count > 0:
            print(f"  - {err_type}: {count}")
    total_errors = sum(errors.values())
    print(f"  总错误数: {total_errors}")
    
    if len(samples) == 0:
        raise ValueError("没有有效样本！请检查数据源。")
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(samples)
    
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
    
    print(f"\n✅ Saved {len(train_df)} training, {len(val_df)} validation samples")
    print(f"   输出目录: {output_dir}")
    
    # 验证数据格式
    print(f"\n=== 数据格式验证 ===")
    print(f"prompt 类型: {type(samples[0]['prompt'])}")
    print(f"prompt 长度: {len(samples[0]['prompt'])} messages")
    print(f"prompt[0] role: {samples[0]['prompt'][0]['role']}")
    print(f"prompt[1] role: {samples[0]['prompt'][1]['role']}")
    print(f"user content 前100字符: {samples[0]['prompt'][1]['content'][:100]}...")
    
    return train_df, val_df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(base_dir, "data/gpt-3-noperception-reflection-1-100agents-240months")
    # 输出到 verl_dataset_small 目录（与配置文件匹配）
    output_dir = os.path.join(base_dir, "data/verl_dataset_small")
    
    prepare_verl_dataset(data_dir, output_dir)
