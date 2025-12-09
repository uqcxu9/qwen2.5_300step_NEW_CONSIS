import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

save_path = './'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

prompt_cost_1k, completion_cost_1k = 0.001, 0.002
QWEN_MODEL = None
QWEN_TOKENIZER = None

def get_qwen_model():
    global QWEN_MODEL, QWEN_TOKENIZER
    if QWEN_MODEL is None:
        print("Loading Qwen2.5-7B-Instruct...")
        QWEN_MODEL = LLM(
            model="/workspace/models/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        from transformers import AutoTokenizer
        QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
            "/workspace/models/Qwen2.5-7B-Instruct",
            trust_remote_code=True
        )
        print("✅ Qwen model loaded!")
    return QWEN_MODEL, QWEN_TOKENIZER

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100, model_type='gpt', seed=42):
    if model_type == 'qwen':
        llm, tokenizer = get_qwen_model()
        
        prompts = []
        for dialog in dialogs:
            prompt = tokenizer.apply_chat_template(
                dialog,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        sampling_params = SamplingParams(
            temperature=max(temperature, 0.01),
            max_tokens=max_tokens,
            top_p=1.0,
            seed=seed, 
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            text = text.replace('```json', '').replace('```', '')
            import re
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                text = json_match.group(0)
            results.append(text.strip())
        
        return results, 0.0
    else:
        from functools import partial
        get_completion_partial = partial(get_completion, temperature=temperature, max_tokens=max_tokens, model_type='gpt', seed=seed)
        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.map(get_completion_partial, dialogs)
        total_cost = sum([cost for _, cost in results])
        return [response for response, _ in results], total_cost


def get_completion(dialogs, temperature=0, max_tokens=100, model_type='gpt', seed=42):
    if model_type == 'qwen':
        llm, tokenizer = get_qwen_model()
        
        prompt = tokenizer.apply_chat_template(
            dialogs,
            tokenize=False,
            add_generation_prompt=True
        )
        
        sampling_params = SamplingParams(
            temperature=max(temperature, 0.01),
            max_tokens=max_tokens,
            top_p=1.0,
            seed=seed, 
        )
        
        outputs = llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text.strip()

        text = text.replace('```json', '').replace('```', '')
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            text = json_match.group(0)
        
        return text.strip(), 0.0
    else:

        import openai
        openai.api_key = 'Your Key'
        import time
        
        max_retries = 20
        for i in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=dialogs,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                this_cost = prompt_tokens/1000*prompt_cost_1k + completion_tokens/1000*completion_cost_1k
                return response.choices[0].message["content"], this_cost
            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(6)
                else:
                    print(f"An error of type {type(e).__name__} occurred: {e}")
                    return "Error", 0.0

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'
# simulate_utils.py 末尾添加

def build_obs_prompt(
    name, age, city, job, offer,
    current_time_str,
    skill, max_l,
    prev_skill,
    consumption,
    tax_paid, lump_sum,
    curr_rates, brackets,
    price, prev_price,
    interest_rate,
    wealth,
    macro_signal="",
    few_shot_examples="",
    tax_model='us-federal-single-filer-2018-scaled'
):
    """
    构造完整的经济决策prompt
    统一在simulate.py和prepare_verl_data.py中使用
    """
    
    problem_prompt = f'''You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings. Now it's {current_time_str}.'''
    
    if job == 'Unemployment':
        job_prompt = f'''In the previous month, you became unemployed and had no income. Now, you are invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}.'''
    else:
        if skill >= prev_skill:
            job_prompt = f'''In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is increased compared to the last month due to the inflation of labor market.'''
        else:
            job_prompt = f'''In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is decreased compared to the last month due to the deflation of labor market.'''
    
    consumption_prompt = f'''Besides, your consumption was ${consumption:.2f}.'''
    
    if tax_model == 'us-federal-single-filer-2018-scaled':
        tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}. In this month, the government sets the brackets: {format_numbers(brackets)} and their corresponding rates: {format_numbers(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
    else:
        tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}. In this month, according to the optimal taxation theory, Saez Tax, the brackets are not changed: {format_numbers(brackets)} but the government has updated corresponding rates: {format_percentages(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
    
    if prev_price is None:
        price_prompt = f'''Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}.'''
    elif price >= prev_price:
        price_prompt = f'''Meanwhile, inflation has led to a price increase in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
    else:
        price_prompt = f'''Meanwhile, deflation has led to a price decrease in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
    
    job_prompt = prettify_document(job_prompt)
    
    obs_prompt = f'''{problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt} Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%.{macro_signal}{few_shot_examples} With all these factors in play, and considering aspects like your living costs, any future aspirations, the broader economic trends, and how your buffer ratio compares to the examples above, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price? **IMPORTANT:** Reply with ONLY a valid JSON object. Required format: {{"work": <number>, "consumption": <number>}} where both are values between 0 and 1 with intervals of 0.02.'''
    
    return prettify_document(obs_prompt)