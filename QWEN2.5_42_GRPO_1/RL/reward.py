import json
import re
import math
import random


def range_reward(x: float, low: float, high: float) -> float:
    """
    改进版：区间内奖励更平缓，避免"随便落在区间里就行"
    - 在区间内：中间最好 (=1.0)，边界 (=0.5)
    - 超出区间：按距离扣分
    
    返回值范围：约 [-1, 1]
    """
    # 除零保护
    width = max(high - low, 1e-6)
    
    if x < low:
        return -min((low - x) / width, 1.0)
    elif x > high:
        return -min((x - high) / width, 1.0)
    else:
        mid = 0.5 * (low + high)
        half = max(0.5 * width, 1e-6)
        return 0.5 + 0.5 * (1.0 - abs(x - mid) / half)



def parse_action(response: str):
    """解析模型输出的 JSON"""
    try:
        text = response.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except:
        return None


def _to_float_or_none(x):
    """将值转换为 float，如果是 None/NaN/Inf 则返回 None"""
    if x is None:
        return None
    try:
        x = float(x)
    except:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


_DEBUG_COUNT = 0
_DEFAULT_VALUE_COUNT = 0  # 追踪默认值触发次数
_SEED_SET = False  # 控制随机种子只设置一次
_FREEZE_STATS = False   # ✅ 新增
_WARMUP_N = 5000        # ✅ 新增
# ===== adaptive thresholds (data-driven, online stats) =====
_RESERVOIR = {}     # key -> list[float]
_RESERVOIR_N = {}   # key -> total seen count (int)
_WORK_MEAN = {}     # regime -> {'n': int, 'mean': float}
_WORK_BR_MEAN = {}  # (regime, bucket) -> {'n': int, 'mean': float}
_RESERVOIR_SIZE = 2000

def _reservoir_update(key, x: float, size=_RESERVOIR_SIZE):
    """Reservoir sampling with correct total-seen counter (unbiased)."""
    if _FREEZE_STATS or x is None:  # ✅ 改这行
        return

    xs = _RESERVOIR.get(key)
    if xs is None:
        xs = []
        _RESERVOIR[key] = xs

    seen = _RESERVOIR_N.get(key, 0) + 1
    _RESERVOIR_N[key] = seen

    if len(xs) < size:
        xs.append(float(x))
        return

    # pick j in [0, seen-1], replace only if j < size
    j = random.randint(0, seen - 1)
    if j < size:
        xs[j] = float(x)

def _quantile_from_reservoir(key, q: float, fallback: float):
    xs = _RESERVOIR.get(key)
    if not xs:
        return float(fallback)
    ys = sorted(xs)
    idx = int(round(q * (len(ys) - 1)))
    idx = max(0, min(len(ys) - 1, idx))
    return float(ys[idx])

def _mean_update(d: dict, x: float) -> dict:
    n = d["n"] + 1
    d["mean"] += (x - d["mean"]) / n
    d["n"] = n
    return d

def _update_gt_work_stats(regime: str, gt_work):
    if _FREEZE_STATS or gt_work is None:  # ✅ 改这行
        return
    x = float(gt_work)
    st = _WORK_MEAN.get(regime)
    if st is None:
        _WORK_MEAN[regime] = {"n": 1, "mean": x}
    else:
        _mean_update(st, x)

def _adaptive_work_target_r(regime: str, fallback: float):
    st = _WORK_MEAN.get(regime)
    if st is None or st["n"] < 50:  # warmup
        return float(fallback)
    return float(max(0.0, min(1.0, st["mean"])))

def _update_gt_work_br_stats(regime: str, br: float, gt_work):
    if _FREEZE_STATS or gt_work is None:  # ✅ 改这行
        return
    b = _br_bucket(br)  # uses adaptive Q20/Q80
    key = (str(regime), b)
    st = _WORK_BR_MEAN.get(key)
    x = float(gt_work)
    if st is None:
        _WORK_BR_MEAN[key] = {"n": 1, "mean": x}
    else:
        _mean_update(st, x)

def _adaptive_work_endpoints(regime: str, br: float, work_min: float,
                             fallback_low: float = 0.90, fallback_high: float = 0.50,
                             warmup_n: int = 50):
    # low endpoint: E[gt_work | bucket=low, regime]
    st_low = _WORK_BR_MEAN.get((str(regime), "low"))
    if st_low is None or st_low["n"] < warmup_n:
        low_ep = float(fallback_low)
    else:
        low_ep = float(st_low["mean"])

    # high endpoint: E[gt_work | bucket=high, regime], then clamp to [work_min, 1]
    st_high = _WORK_BR_MEAN.get((str(regime), "high"))
    if st_high is None or st_high["n"] < warmup_n:
        high_ep = float(fallback_high)
    else:
        high_ep = float(st_high["mean"])

    low_ep  = max(0.0, min(1.0, low_ep))
    high_ep = max(work_min, min(1.0, high_ep))

    # enforce monotonic: low buffer should imply >= work than high buffer
    if low_ep < high_ep:
        # swap to ensure low_ep >= high_ep
        low_ep, high_ep = high_ep, low_ep

    return low_ep, high_ep

# ===== adaptive consumption target (data-driven, online stats) =====
_CONS_STATS = {}  # key -> {'n': int, 'mean': float, 'm2': float}

def _br_bucket(br: float) -> str:
    # use the same adaptive thresholds as work part: Q20/Q80
    br_lo = _quantile_from_reservoir(("br", "all"), 0.20, fallback=1.6)
    br_hi = _quantile_from_reservoir(("br", "all"), 0.80, fallback=4.2)
    if br_hi <= br_lo + 1e-6:
        br_hi = br_lo + 1.0

    if br <= br_lo:
        return "low"
    elif br >= br_hi:
        return "high"
    else:
        return "mid"

def _welford_update(stat: dict, x: float) -> dict:
    # Welford online update for mean/std
    n = stat["n"] + 1
    delta = x - stat["mean"]
    mean = stat["mean"] + delta / n
    m2 = stat["m2"] + delta * (x - mean)
    stat["n"] = n
    stat["mean"] = mean
    stat["m2"] = m2
    return stat

def _adaptive_cons_target(regime: str, br: float, gt_c):

    key = (str(regime), _br_bucket(br))

    # update stats using gt_consumption when available
    if gt_c is not None and not _FREEZE_STATS:
        x = float(gt_c)
        st = _CONS_STATS.get(key)
        if st is None:
            st = {"n": 1, "mean": x, "m2": 0.0}
            _CONS_STATS[key] = st
        else:
            _welford_update(st, x)

    st = _CONS_STATS.get(key)
    if st is None:
        # warm start: use global gt_consumption quantiles (data-driven), fallback only if empty
        c_lo = _quantile_from_reservoir(("gt_c", "all"), 0.35, fallback=0.50)
        c_hi = _quantile_from_reservoir(("gt_c", "all"), 0.65, fallback=0.70)
        target = 0.5 * (c_lo + c_hi)
        band = max(0.10, min(0.25, 0.5 * (c_hi - c_lo)))
        return float(max(0.05, min(0.95, target))), float(band)

    n = st["n"]
    var = st["m2"] / max(n - 1, 1)
    std = math.sqrt(max(var, 1e-6))

    target = max(0.05, min(0.95, float(st["mean"])))
    band = max(0.10, min(0.25, 1.5 * std))  # adaptive tolerance
    return target, band


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float:
    global _DEBUG_COUNT, _DEFAULT_VALUE_COUNT, _SEED_SET, _FREEZE_STATS
    if not _SEED_SET:
        random.seed(42)
        _SEED_SET = True
    _DEBUG_COUNT += 1
    # ✅ 新增：warmup 后冻结
    if _DEBUG_COUNT >= _WARMUP_N and not _FREEZE_STATS:
        _FREEZE_STATS = True
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(f"[INFO] Stats frozen at sample {_DEBUG_COUNT}\n")
        except:
            pass
    # 调试日志（只记录前 5 个）
    if _DEBUG_COUNT <= 5:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(f"\n=== DEBUG {_DEBUG_COUNT} ===\n")
                f.write(f"solution_str: {repr(solution_str)[:500]}\n")
                f.write(f"extra_info: {repr(extra_info)[:300]}\n")
        except:
            pass

    if data_source != "econ_agent":
        return 0.0

    reward = 0.0

    # ========== 1. JSON 格式检查 ==========
    action = parse_action(solution_str)
    if action is None:
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

    # ========== 2. 范围检查（软约束）==========
    if not (0 <= work <= 1):
        reward -= 0.3
    if not (0 <= consumption <= 1):
        reward -= 0.3

    work = max(0.0, min(1.0, work))
    consumption = max(0.0, min(1.0, consumption))

    # ========== 3. 解析 extra_info ==========
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    elif extra_info is None:
        extra_info = {}

    # ------ 3.1 微观变量 ------
    income   = _to_float_or_none(extra_info.get('income', 0))   or 0.0
    lump_sum = _to_float_or_none(extra_info.get('lump_sum', 0)) or 0.0
    tax_paid = _to_float_or_none(extra_info.get('tax_paid', 0)) or 0.0
    wealth   = _to_float_or_none(extra_info.get('wealth', 0))   or 0.0

    dpi = _to_float_or_none(extra_info.get('dpi', None))
    if dpi is None:
        dpi = income + lump_sum - tax_paid
    dpi = float(dpi)

    # amount：用真实 dpi（clamp 到非负）
    dpi_amt = max(dpi, 0.0)

    buffer_ratio = _to_float_or_none(extra_info.get('buffer_ratio', None))
    if buffer_ratio is None:
        if dpi > 1e-6:
            cash_on_hand = wealth + dpi
            buffer_ratio = cash_on_hand / (dpi + 1e-8)
        else:
            buffer_ratio = 1.0
    buffer_ratio = max(0.0, min(10.0, float(buffer_ratio)))



    # ------ 3.2 宏观变量 ------
    unemp = _to_float_or_none(extra_info.get('unemployment_rate', None))
    gdp_g = _to_float_or_none(extra_info.get('gdp_growth', None))
    infl  = _to_float_or_none(extra_info.get('price_inflation', None))

    # ✅ 提前解析 regime + regime_strength，让 5.2 可按宏观强度调权/融合目标
    regime = extra_info.get("regime", None)
    if regime is None:
        regime = "normal"
        _DEFAULT_VALUE_COUNT += 1
        if _DEFAULT_VALUE_COUNT <= 10:
            try:
                with open('/workspace/reward_debug.log', 'a') as f:
                    f.write(f"[WARNING] Missing regime at sample {_DEBUG_COUNT}\n")
            except:
                pass

    regime_strength = _to_float_or_none(extra_info.get("regime_strength", None))
    if regime_strength is None:
        regime_strength = 0.15
        _DEFAULT_VALUE_COUNT += 1
        if _DEFAULT_VALUE_COUNT <= 10:
            try:
                with open('/workspace/reward_debug.log', 'a') as f:
                    f.write(f"[WARNING] Missing regime_strength at sample {_DEBUG_COUNT}\n")
            except:
                pass

    regime_strength = max(0.0, min(1.0, regime_strength))
    # ========== 4. Saving Rate ==========
    saving_rate = 1.0 - consumption

    # ========== 5. 评分计算 ==========
    # ------ 5.1 储蓄率约束 ------
    sr_reward = range_reward(saving_rate, 0.014, 0.60)

    # ------ 5.2 Work 行为偏好（权重 0.25）------



    # ✅ 连续目标：buffer_ratio 越大，work_target 越低（才能让相关性显著变负）
    br = buffer_ratio

    # ---- update online stats (use gt_*) ----
    gt_work = _to_float_or_none(extra_info.get("gt_work", None))
    _update_gt_work_stats(regime, gt_work)

    # buffer_ratio quantiles FIRST (so _br_bucket uses updated Q20/Q80)
    _reservoir_update(("br", "all"), br)
    _update_gt_work_br_stats(regime, br, gt_work)   # conditional mean by br bucket

    # gt_work distribution quantiles for boundary guard (global)
    if gt_work is not None:
        _reservoir_update(("gt_work", "all"), gt_work)
    # ✅ 常量定义
    WORK_MIN = _quantile_from_reservoir(("gt_work", "all"), 0.10, fallback=0.50)
    WORK_MIN = max(0.0, min(1.0, WORK_MIN))

    # adaptive br thresholds from quantiles
    br_lo = _quantile_from_reservoir(("br", "all"), 0.20, fallback=1.6)
    br_hi = _quantile_from_reservoir(("br", "all"), 0.80, fallback=4.2)
    br_lo, br_hi = (br_lo, br_hi) if br_hi > br_lo + 1e-6 else (br_lo, br_lo + 1.0)

    # endpoints: data-driven from gt_work conditional means
    low_ep, high_ep = _adaptive_work_endpoints(
        regime=regime,
        br=br,
        work_min=WORK_MIN,
        fallback_low=0.90,
        fallback_high=WORK_MIN,
        warmup_n=50,
    )

    # target: br <= Q20 -> low_ep, br >= Q80 -> high_ep, interpolate in between
    if br <= br_lo:
        work_target = low_ep
    elif br >= br_hi:
        work_target = high_ep
    else:
        work_target = low_ep + (high_ep - low_ep) * (br - br_lo) / (br_hi - br_lo)

    # band: use mid-quantiles for "central" range (replace 1.8/3.8)
    br_mid_lo = _quantile_from_reservoir(("br", "all"), 0.30, fallback=1.8)
    br_mid_hi = _quantile_from_reservoir(("br", "all"), 0.70, fallback=3.8)
    work_band_br = 0.12 if (br_mid_lo <= br <= br_mid_hi) else 0.10

    # ✅ regime 侧 work target（用于和 buffer_ratio target 融合，避免 5.2/5.3 冲突）
    # fallback keeps your original intent during warmup, then becomes data-driven
    if regime == "recession":
        work_target_r = _adaptive_work_target_r("recession", fallback=0.85)
    elif regime == "boom":
        work_target_r = _adaptive_work_target_r("boom", fallback=0.30)
    else:
        work_target_r = _adaptive_work_target_r("normal", fallback=0.55)

    # ✅ 融合系数：宏观越"强"（boom/recession 越明显），越听 regime
    # alpha in [0, 0.6], grows with regime_strength (floor-free)
    # dataset: normal ~0.15, boom/recession >=0.2; here we set alpha=0 in normal
    if regime in ("boom", "recession"):
        alpha = 0.6 * max(0.0, min(1.0, regime_strength))
    else:
        alpha = 0.0

    work_target_mix = (1.0 - alpha) * work_target + alpha * work_target_r

    work_reward = range_reward(
        work,
        max(0.0, work_target_mix - work_band_br),
        min(1.0, work_target_mix + work_band_br),
    )
    if work > work_target_mix:
        excess = (work - work_target_mix) / max(work_band_br, 1e-6)  # normalized
        work_reward -= 0.20 * min(excess ** 1.3, 3.0) 


    # ✅ 双层 Barrier：经济合理 + 破0.80盆地
    overwork_pen = 0.0

    BARRIER_OFFSET = work_band_br
    BARRIER_K_MAIN = 0.08
   

    thr_follow = work_target_mix + BARRIER_OFFSET
    barrier_threshold = min(0.95, thr_follow)
    _dbg_barrier = barrier_threshold

    if work > barrier_threshold + 1e-6:
        overwork_pen -= BARRIER_K_MAIN * min((work - barrier_threshold) / 0.15, 1.0)

    # Layer 2: global anti-boundary penalty (only when high work is NOT economically justified)
    # adaptive boundary start from gt_work distribution (Q95), fallback=0.92
    GLOBAL_START = _quantile_from_reservoir(("gt_work", "all"), 0.95, fallback=0.92)
    GLOBAL_START = max(0.80, min(0.97, GLOBAL_START))  # keep sane range
    GLOBAL_SPAN  = 0.10
    BARRIER_K_GLOBAL = 0.02

    layer2_pen = 0.0

    # 只有当目标本身不要求很高劳动时，才防止贴边
    if work_target_mix < 0.85:
        if work > GLOBAL_START + 1e-6:
            layer2_pen -= BARRIER_K_GLOBAL * min((work - GLOBAL_START) / GLOBAL_SPAN, 1.0)

    overwork_pen += layer2_pen

    # ✅ 独立过度消费惩罚：防止 consumption 锁死到极端值
    overconsume_pen = 0.0
    if consumption > 0.90:
        overconsume_pen -= 0.20 * (consumption - 0.90) / 0.10  # 0.90->0, 1.00->-0.20

    # ===== 5.3(A) consumption target: data-driven adaptive =====

    # data-driven consumption target (adaptive from gt_consumption stats)
    gt_c = _to_float_or_none(extra_info.get("gt_consumption", None))
    if gt_c is not None:
        _reservoir_update(("gt_c", "all"), gt_c)
    cons_target_mix, cons_band = _adaptive_cons_target(regime, br, gt_c)


    cons_r = range_reward(
        consumption,
        max(0.0, cons_target_mix - cons_band),
        min(1.0, cons_target_mix + cons_band),
    )

    # action_struct 只用消费信号
    action_struct = cons_r


    # (B) anti-extreme brake（保留原逻辑；注意它会乘 regime_strength）
    extreme_pen = 0.0
    if work < 0.05 or work > 0.95:
        extreme_pen -= 0.10
    if consumption < 0.05 or consumption > 0.95:
        extreme_pen -= 0.10


    # (C) optional macro guardrail
    guard_parts = []
    guard_w = []

    if unemp is not None:
        guard_parts.append(range_reward(unemp, 0.02, 0.20))
        guard_w.append(0.40)
    if gdp_g is not None:
        guard_parts.append(range_reward(gdp_g, -5.0, 10.0))
        guard_w.append(0.35)
    if infl is not None:
        guard_parts.append(range_reward(infl, -2.0, 8.0))
        guard_w.append(0.25)

    if guard_w:
        wsum = sum(guard_w)
        guard = sum(p * w for p, w in zip(guard_parts, guard_w)) / wsum
    else:
        guard = 0.0

    guard = max(-1.0, min(1.0, guard))

    # (D) combine
    macro_reward = (0.9 * action_struct + 0.1 * guard + extreme_pen)

    # ========== normalized mixture (weights sum to 1) ==========
    W_SR = 0.10
    W_WORK = 0.35
    W_MACRO = 0.45
    W_PEN = 0.10  # penalty channel

    w_sr = W_SR
    w_work = W_WORK * (1.0 - 0.6 * alpha)
    w_macro = W_MACRO * regime_strength


    # map penalties into [-1, 0] so it can be mixed as a term
    penalty_raw = overwork_pen + overconsume_pen  # about [-0.30, 0]
    penalty_term = max(-1.0, min(0.0, penalty_raw / 0.30))

    w_pen = W_PEN

    w_sum = w_sr + w_work + w_macro + w_pen + 1e-8

    reward += (w_sr / w_sum) * sr_reward
    reward += (w_work / w_sum) * work_reward
    reward += (w_macro / w_sum) * macro_reward
    reward += (w_pen / w_sum) * penalty_term

    # ========== 6. 调试日志 ==========
    if _DEBUG_COUNT <= 20:
        try:
            unemp_str = f"{unemp:.3f}" if unemp is not None else "None"
            gdp_str = f"{gdp_g:.2f}" if gdp_g is not None else "None"
            infl_str = f"{infl:.2f}" if infl is not None else "None"
            macro_used = ''.join([
                'pi' if infl is not None else '',
                'u'  if unemp is not None else '',
                'g'  if gdp_g is not None else ''
            ]) or 'none'
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(
                    f"[{_DEBUG_COUNT:03d}] "
                    f"sr={saving_rate:.3f} sr_r={sr_reward:.3f} sr_contrib={((w_sr/w_sum)*sr_reward):.3f} | "
                    f"buf={buffer_ratio:.2f} work={work:.2f} "
                    f"tgt={work_target_mix:.2f} bar={_dbg_barrier:.2f} "
                    f"wr={work_reward:.2f} work_w={(w_work/w_sum):.3f} work_contrib={((w_work/w_sum)*work_reward):.3f} | "
                    f"pen_raw={penalty_raw:.3f} pen_term={penalty_term:.3f} pen_contrib={((w_pen/w_sum)*penalty_term):.3f} | "
                    f"macro={macro_reward:.3f} macro_w={(w_macro/w_sum):.3f} macro_contrib={((w_macro/w_sum)*macro_reward):.3f} | "
                    f"total={reward:.3f}\n"
                )
        except:
            pass

    # 每 1000 个样本报告一次默认值触发情况
    if _DEBUG_COUNT % 1000 == 0 and _DEFAULT_VALUE_COUNT > 0:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(
                    f"[STATS] Processed {_DEBUG_COUNT} samples, default value triggered {_DEFAULT_VALUE_COUNT} times "
                    f"({_DEFAULT_VALUE_COUNT/_DEBUG_COUNT*100:.2f}%)\n"
                )
        except:
            pass

    # ========== 7. 最终 clip（数值稳定）==========
    reward = max(-1.0, min(1.0, reward))
    return reward