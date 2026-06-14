#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无图形界面时使用
import matplotlib.pyplot as plt
from rpg import MagicItemDistribution, DamageDistribution

# 忽略警告（可选）
import warnings
warnings.filterwarnings('ignore')

print("=== RPG 伤害分析系统 ===\n")

# 初始化
bonus_probs = np.array([0.0, 0.55, 0.25, 0.12, 0.06, 0.02])
stats_probs = np.ones(6) / 6.0
rso = np.random.RandomState(234892)

item_dist = MagicItemDistribution(bonus_probs, stats_probs, rso=rso)

print("随机魔法物品示例:")
for i in range(5):
    print(f"  {i+1}: {item_dist.sample()}")

# 概率计算
item = item_dist.sample()
print(f"\n随机物品: {item}")
print(f"log-PMF: {item_dist.log_pmf(item)}")
print(f"PMF: {item_dist.pmf(item)}")

# 伤害模拟
print("\n=== 伤害分布模拟 ===")
damage_dist = DamageDistribution(2, item_dist, num_hits=3, rso=rso)

print("采样 100,000 次...")
samples = np.array([damage_dist.sample() for i in range(100000)])

minval = samples.min()
maxval = samples.max()
median = np.percentile(samples, 50)

print(f"\n统计结果:")
print(f"  最小值: {minval}")
print(f"  最大值: {maxval}")
print(f"  中位数: {median}")
print(f"  平均值: {samples.mean():.2f}")
print(f"  标准差: {samples.std():.2f}")

# 绘制图表
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=samples.max() + 1, range=(0, samples.max()), 
         histtype='step', color='k')
plt.vlines([minval, maxval, median], 0, 4000, 
           color='r', linestyle='--', linewidth=2)
plt.xlabel("Damage")
plt.ylabel("Number of samples")
plt.title("Distribution over attack damage for 2 weapons")
plt.tight_layout()
plt.savefig("damage_distribution.png", dpi=150)

print("\n图表已保存: damage_distribution.png")
print("完成!")