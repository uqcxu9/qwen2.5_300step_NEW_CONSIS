加了import pickle as pkl
import numpy as np
import pandas as pd

# ========== 1. 加载数据 ==========
data_path = r'/workspace/QWEN2.5/data/gpt-3-noperception-reflection-1-100agents-240months'

with open(f'{data_path}/dense_log.pkl', 'rb') as f:
    dense_log = pkl.load(f)

states = dense_log['states']
actions = dense_log['actions']
periodic_tax = dense_log['PeriodicTax']

print(f"总时间步数: {len(states)}")
print(f"actions长度: {len(actions)}")
print(f"periodic_tax长度: {len(periodic_tax)}")

# ========== 2. 获取价格数据 ==========
class DummyUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if 'ai_economist' in module:
            return type(name, (), {})
        return super().find_class(module, name)

env_file = f'{data_path}/env_240.pkl'
with open(env_file, "rb") as f:
    env = DummyUnpickler(f).load()

prices = list(env.world.price)
print(f"价格数据长度: {len(prices)}")

# ========== 辅助函数 ==========
A = 1  # 生产率
num_labor_hours = 168  # 月工作小时数

def calculate_dpi(t, agent_id_str, states, periodic_tax):
    """计算DPI = income + lump_sum - tax_paid"""
    income = states[t][agent_id_str]['income']['Coin']
    lump_sum = periodic_tax[t].get(agent_id_str, {}).get('lump_sum', 0)
    tax_paid = periodic_tax[t].get(agent_id_str, {}).get('tax_paid', 0)
    return income + lump_sum - tax_paid

def calculate_monthly_gdp(t, states, actions, prices):
    """计算月度GDP = S × P"""
    monthly_supply = 0
    for agent_id, action in actions[t].items():
        if agent_id == 'p':
            continue
        
        # 提取工作决策
        if isinstance(action, dict):
            labor = int(action.get('SimpleLabor', 0))
        elif isinstance(action, (list, tuple)) and len(action) >= 1:
            labor = int(action[0])
        else:
            labor = 0
        
        monthly_supply += labor * num_labor_hours * A
    
    return monthly_supply * prices[t]

# ========== 3. 计算宏观指标（改为年度） ==========
# ========== 3. 计算宏观指标（月度失业率 + 年度GDP/通胀） ==========
print("\n📊 Step 1: 筛选宏观表现好的月份...")

max_t = min(len(states), len(periodic_tax), len(actions), len(prices))
macro_good_months = []  # 改：存储月份索引，不是年份

for year in range(2, 21):
    year_start_month = (year - 1) * 12
    year_end_month = year * 12
    
    if year_end_month > max_t:
        break
    
    # === 年度GDP增长（不变） ===
    curr_year_gdp = sum(calculate_monthly_gdp(t, states, actions, prices) 
                       for t in range(year_start_month, year_end_month))
    prev_year_gdp = sum(calculate_monthly_gdp(t, states, actions, prices) 
                       for t in range(year_start_month-12, year_start_month))
    
    if prev_year_gdp > 0:
        gdp_growth = (curr_year_gdp - prev_year_gdp) / prev_year_gdp * 100
    else:
        continue
    
    # === 年度通胀率（不变） ===
    curr_avg_price = np.mean([prices[t] for t in range(year_start_month, year_end_month)])
    prev_avg_price = np.mean([prices[t] for t in range(year_start_month-12, year_start_month)])
    inflation = (curr_avg_price - prev_avg_price) / prev_avg_price * 100
    
    # === 年度宏观约束（改：只检查GDP和通胀） ===
    gdp_good = (-2 <= gdp_growth <= 11)
    inflation_good = (-0.36 <= inflation <= 4.7)
    year_macro_good = gdp_good and inflation_good
    
    # === 🆕 新增：遍历该年每个月，检查月度失业率 ===
    for t in range(year_start_month, year_end_month):
        # 月度失业率
        unemployed = 0
        employed = 0
        for aid, state in states[t].items():
            if aid == "p" or not isinstance(state, dict):
                continue
            job = state.get("endogenous", {}).get("job")
            if job == "Unemployment":
                unemployed += 1
            else:
                employed += 1
        
        labor_force = employed + unemployed
        monthly_unemployment = unemployed / labor_force if labor_force > 0 else 0
        
        # 月度失业率约束
        unemployment_good = (0.035 <= monthly_unemployment <= 0.148)
        
        # 该月宏观好 = 年度GDP/通胀好 + 月度失业率好
        if year_macro_good and unemployment_good:
            macro_good_months.append(t)

print(f"✅ 找到 {len(macro_good_months)} 个宏观表现好的月份")

# ========== 4. 从宏观好年份中提取微观好决策 ==========
# ========== 4. 从宏观好月份中提取微观好决策 ==========
print("\n📊 Step 2: 从宏观好月份中提取微观好决策...")

good_decisions = []

# 🆕 首先计算每个agent每年的MPC（预处理）
agent_year_mpc = {}  # {(agent_id, year): mpc}

for year in range(2, 21):
    year_start_month = (year - 1) * 12
    year_end_month = year * 12
    
    if year_end_month > max_t:
        break
    
    for agent_id in range(100):
        agent_id_str = str(agent_id)
        
        yearly_dpi_change = 0
        yearly_c_change = 0
        
        for t in range(year_start_month, year_end_month):
            if t == 0 or agent_id_str not in states[t]:
                continue
            
            curr_dpi = calculate_dpi(t, agent_id_str, states, periodic_tax)
            prev_dpi = calculate_dpi(t-1, agent_id_str, states, periodic_tax)
            
            curr_c = states[t][agent_id_str]['consumption']['Coin']
            prev_c = states[t-1][agent_id_str]['consumption']['Coin']
            
            yearly_dpi_change += (curr_dpi - prev_dpi)
            yearly_c_change += (curr_c - prev_c)
        
        if abs(yearly_dpi_change) > 500:
            agent_year_mpc[(agent_id, year)] = yearly_c_change / yearly_dpi_change
        else:
            agent_year_mpc[(agent_id, year)] = None

# 🆕 遍历宏观好月份（不是年份）
for t in macro_good_months:
    if t == 0:
        continue
    
    # 🆕 确定当前月份属于哪一年
    current_year = (t // 12) + 1
    
    for agent_id in range(100):
        agent_id_str = str(agent_id)
        
        if agent_id_str not in states[t]:
            continue
        
        # === 提取月度数据（不变） ===
        curr_consumption = states[t][agent_id_str]['consumption']['Coin']
        prev_consumption = states[t-1][agent_id_str]['consumption']['Coin']
        curr_income = states[t][agent_id_str]['income']['Coin']
        prev_income = states[t-1][agent_id_str]['income']['Coin']
        curr_wealth = states[t][agent_id_str]['inventory']['Coin']
        
        curr_tax = periodic_tax[t].get(agent_id_str, {}).get('tax_paid', 0)
        prev_tax = periodic_tax[t-1].get(agent_id_str, {}).get('tax_paid', 0)
        curr_lump = periodic_tax[t].get(agent_id_str, {}).get('lump_sum', 0)
        prev_lump = periodic_tax[t-1].get(agent_id_str, {}).get('lump_sum', 0)
        
        curr_dpi = curr_income + curr_lump - curr_tax
        prev_dpi = prev_income + prev_lump - prev_tax
        
        # === 月度物理约束（不变） ===
        if curr_consumption > curr_wealth + curr_income + 100:
            continue
        
        # === 月度储蓄率约束（不变） ===
        if curr_dpi > 50:
            saving_rate = (curr_dpi - curr_consumption) / curr_dpi
            if saving_rate < -0.2 or saving_rate > 0.9:
                continue
        
        # === 年度MPC约束（改：用预处理的数据） ===
        yearly_mpc = agent_year_mpc.get((agent_id, current_year), None)
        if yearly_mpc is not None:
            if yearly_mpc < -1.0 or yearly_mpc > 2.0:
                continue
        
        # === 提取决策（不变） ===
        job = states[t][agent_id_str].get('endogenous', {}).get('job')
        work_decision = 0.0 if job == "Unemployment" else 1.0
        
        if agent_id_str in actions[t]:
            action_data = actions[t][agent_id_str]
            if isinstance(action_data, dict):
                consumption_idx = action_data.get('SimpleConsumption', 25)
            elif isinstance(action_data, (list, tuple)) and len(action_data) >= 2:
                consumption_idx = action_data[1]
            else:
                consumption_idx = 25
            consumption_prop = consumption_idx * 0.02
        else:
            consumption_prop = 0.5
        
        # === 计算score（不变） ===
        score = 0
        score += 10
        if yearly_mpc is not None:
            if 0.05 <= yearly_mpc <= 0.9:
                score += 5
            elif 0.0 <= yearly_mpc <= 1.2:
                score += 2
        if curr_dpi > 50:
            saving_rate = (curr_dpi - curr_consumption) / curr_dpi
            if 0.014 <= saving_rate <= 0.318:
                score += 3
            elif 0.0 <= saving_rate <= 0.5:
                score += 1
        if curr_consumption <= curr_wealth + curr_income:
            score += 1
        
        # === 保存决策（改：year改为current_year） ===
        good_decisions.append({
            'timestep': t,
            'year': current_year,
            'agent_id': agent_id,
            'prev_consumption': prev_consumption,
            'curr_consumption': curr_consumption,
            'prev_income': prev_income,
            'curr_income': curr_income,
            'curr_wealth': curr_wealth,
            'prev_dpi': prev_dpi,
            'curr_dpi': curr_dpi,
            'work_decision': work_decision,
            'consumption_prop': consumption_prop,
            'is_macro_good': True,
            'yearly_mpc': yearly_mpc if yearly_mpc is not None else 0.0,
            'score': score,
        })

print(f"✅ 提取了 {len(good_decisions)} 个好决策")
if len(good_decisions) > 0:
    print(f"占总决策的 {len(good_decisions)/(max_t*100)*100:.2f}%")

# ========== 5. 保存 ==========
if len(good_decisions) == 0:
    print("\n⚠️ 警告：没有找到符合条件的决策！")
    print("建议：")
    print("1. 检查宏观约束是否过严")
    print("2. 放宽年度MPC范围（如[-2, 3]）")
    print("3. 检查数据是否正确加载")
else:
    df_good = pd.DataFrame(good_decisions)

    # ========== 🆕 强制就业/失业平衡采样 ==========
    print("\n📊 执行就业/失业平衡采样...")
    
    employed_decisions = df_good[df_good['work_decision'] == 1.0].copy()
    unemployed_decisions = df_good[df_good['work_decision'] == 0.0].copy()
    
    print(f"原始数据：就业 {len(employed_decisions)} 个，失业 {len(unemployed_decisions)} 个")
    print(f"原始失业率：{len(unemployed_decisions)/len(df_good)*100:.1f}%")
    
    # 🎯 目标：15%失业（接近现实的3倍，足够学习）
    target_unemployed_ratio = 0.15
    
    if len(unemployed_decisions) > 0 and len(employed_decisions) > 0:
        original_ratio = len(unemployed_decisions) / len(df_good)
        
        if original_ratio >= target_unemployed_ratio:
            print(f"✅ 失业率已满足目标 ({original_ratio*100:.1f}% >= {target_unemployed_ratio*100:.1f}%)")
        else:
            # 计算需要保留多少就业决策
            target_employed_count = int(len(unemployed_decisions) / target_unemployed_ratio * (1 - target_unemployed_ratio))
            
            if len(employed_decisions) > target_employed_count:
                # 分层采样：50%高分 + 50%随机（保持多样性）
                n_high = int(target_employed_count * 0.5)
                employed_high = employed_decisions.nlargest(n_high, 'score')
                employed_rest = employed_decisions[~employed_decisions.index.isin(employed_high.index)]
                employed_random = employed_rest.sample(n=target_employed_count - n_high, random_state=42)
                
                employed_sampled = pd.concat([employed_high, employed_random])
            else:
                employed_sampled = employed_decisions
            
            # 合并
            df_good = pd.concat([employed_sampled, unemployed_decisions], ignore_index=True)
            
            print(f"✅ 平衡后：就业 {len(employed_sampled)} 个，失业 {len(unemployed_decisions)} 个")
            print(f"   失业占比：{len(unemployed_decisions)/len(df_good)*100:.1f}%")
    else:
        print("⚠️ 无法平衡：缺少就业或失业决策")
    # ========== 平衡采样结束 ==========

    # 统计信息
    print("\n📊 数据质量检查:")
    print(f"来源年份: {sorted(df_good['year'].unique())}")
    print(f"总决策数: {len(df_good)}")
    print(f"就业比例: {(df_good['work_decision']==1).sum()/len(df_good)*100:.1f}%")
    print(f"失业比例: {(df_good['work_decision']==0).sum()/len(df_good)*100:.1f}%")
    print(f"平均分数: {df_good['score'].mean():.2f}")
    
    # 年度MPC统计
    mpc_valid = df_good[df_good['yearly_mpc'] != 0.0]['yearly_mpc']
    if len(mpc_valid) > 0:
        print(f"\n年度MPC统计:")
        print(f"  样本数: {len(mpc_valid)}")
        print(f"  均值: {mpc_valid.mean():.2f}")
        print(f"  中位数: {mpc_valid.median():.2f}")
        print(f"  范围: [{mpc_valid.min():.2f}, {mpc_valid.max():.2f}]")
        print(f"  在[0.05,0.9]范围内: {((mpc_valid>=0.05)&(mpc_valid<=0.9)).sum()/len(mpc_valid)*100:.1f}%")
    
    output_path = f'{data_path}/good_decisions.csv'
    df_good.to_csv(output_path, index=False)
    
    print(f"\n💾 结果已保存到: {output_path}")
    print("\n📊 样本统计:")
    print(df_good[['prev_consumption', 'curr_consumption', 'curr_income', 'curr_wealth', 'yearly_mpc', 'score']].describe())


    注意constraint和路径