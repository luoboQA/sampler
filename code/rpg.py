import numpy as np
from multinomial import MultinomialDistribution


class MagicItemDistribution(object):
    """魔法物品属性加成分布类"""

    # 所有魔法物品的属性名称（顺序固定）
    stats_names = ("dexterity", "constitution", "strength",
                   "intelligence", "wisdom", "charisma")

    def __init__(self, bonus_probs, stats_probs, rso=np.random):
        """初始化魔法物品分布

        Parameters
        ----------
        bonus_probs : numpy array, shape (m,)
            总加成的概率分布，索引 i 对应 +i 加成
        stats_probs : numpy array, shape (6,)
            加成点数在各属性上的分布概率
        rso : RandomState
            随机数生成器
        """
        self.bonus_dist = MultinomialDistribution(bonus_probs, rso=rso)
        self.stats_dist = MultinomialDistribution(stats_probs, rso=rso)

    def sample(self):
        """随机生成一个魔法物品的属性加成

        Returns
        -------
        dict
            属性名 -> 加成点数
        """
        stats = self._sample_stats()
        item_stats = dict(zip(self.stats_names, stats))
        return item_stats

    def log_pmf(self, item):
        """计算给定魔法物品的对数概率"""
        stats = np.array([item[stat] for stat in self.stats_names])
        log_pmf = self._stats_log_pmf(stats)
        return log_pmf

    def pmf(self, item):
        """计算给定魔法物品的概率"""
        return np.exp(self.log_pmf(item))

    def _sample_bonus(self):
        """采样总加成值（+0, +1, ...）"""
        # 采样一次多项分布（只有 1 次事件）
        sample = self.bonus_dist.sample(1)
        # sample 是一个 one‑hot 向量，找到 1 的位置
        bonus = np.argmax(sample)
        return bonus

    def _sample_stats(self):
        """采样总加成点数，并将其分配到各属性上"""
        bonus = self._sample_bonus()             # 先得到总加成
        stats = self.stats_dist.sample(bonus)    # 再分配加成点
        return stats

    def _bonus_log_pmf(self, bonus):
        """计算总加成值的对数概率"""
        if bonus < 0 or bonus >= len(self.bonus_dist.p):
            return -np.inf
        x = np.zeros(len(self.bonus_dist.p))
        x[bonus] = 1
        return self.bonus_dist.log_pmf(x)

    def _stats_log_pmf(self, stats):
        """计算给定属性加成向量的对数概率（包括总加成的概率）"""
        total_bonus = np.sum(stats)                     # 总加成点数
        logp_bonus = self._bonus_log_pmf(total_bonus)   # 总加成的概率
        logp_stats = self.stats_dist.log_pmf(stats)     # 分配方式的概率
        log_pmf = logp_bonus + logp_stats               # 联合概率
        return log_pmf


class DamageDistribution(object):
    """伤害分布类，模拟多轮攻击的总伤害"""

    def __init__(self, num_items, item_dist,
                 num_dice_sides=12, num_hits=1, rso=np.random):
        """初始化伤害分布

        Parameters
        ----------
        num_items : int
            玩家拥有的魔法物品数量
        item_dist : MagicItemDistribution
            魔法物品的分布
        num_dice_sides : int
            每个骰子的面数（默认 12）
        num_hits : int
            攻击命中次数（默认 1）
        rso : RandomState
            随机数生成器
        """
        self.dice_sides = np.arange(1, num_dice_sides + 1)  # 1,2,...,12
        # 多项分布，每个面概率相等
        self.dice_dist = MultinomialDistribution(
            np.ones(num_dice_sides) / float(num_dice_sides), rso=rso)

        self.num_hits = num_hits
        self.num_items = num_items
        self.item_dist = item_dist

    def sample(self):
        """采样一次攻击的总伤害值

        Returns
        -------
        int
            总伤害
        """
        # 1. 生成 num_items 个魔法物品
        items = [self.item_dist.sample() for _ in range(self.num_items)]

        # 2. 基础 1 个骰子 + 所有物品的力量加成之和
        num_dice = 1 + np.sum([item['strength'] for item in items])

        # 3. 投掷骰子（每个命中次数都要投这么多骰子）
        dice_rolls = self.dice_dist.sample(self.num_hits * num_dice)

        # 4. 计算伤害：每个骰子面值 × 出现次数
        damage = np.sum(self.dice_sides * dice_rolls)
        return damage