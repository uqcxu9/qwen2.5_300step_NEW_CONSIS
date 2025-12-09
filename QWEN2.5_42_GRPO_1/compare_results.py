# check_dense_log_keys.py
import pickle as pkl

# 改成你的路径
dense_log_path = "./data/gpt-3-noperception-reflection-1-100agents-240months/dense_log.pkl"

with open(dense_log_path, 'rb') as f:
    dense_log = pkl.load(f)

print("=== dense_log 的所有 keys ===")
print(list(dense_log.keys()))

print("\n=== 查找包含 'tax' 或 'Tax' 的 key ===")
for key in dense_log.keys():
    if 'tax' in key.lower():
        print(f"  Found: '{key}'")

# 如果找到了正确的key，检查其结构
tax_key = None
for key in dense_log.keys():
    if 'tax' in key.lower():
        tax_key = key
        break

if tax_key:
    print(f"\n=== '{tax_key}' 的结构 ===")
    tax_data = dense_log[tax_key]
    print(f"类型: {type(tax_data)}")
    print(f"长度: {len(tax_data)}")
    
    # 看第一个时间步的结构
    print(f"\n第一个时间步 (t=0) 的内容:")
    print(f"  类型: {type(tax_data[0])}")
    if isinstance(tax_data[0], dict):
        print(f"  Keys: {list(tax_data[0].keys())[:5]}...")  # 前5个key
        
        # 看一个agent的数据
        first_agent_key = list(tax_data[0].keys())[0]
        print(f"\n  Agent '{first_agent_key}' 的数据:")
        print(f"    {tax_data[0][first_agent_key]}")
else:
    print("\n❌ 没有找到包含 'tax' 的 key!")
    print("请检查 dense_log 的完整结构")