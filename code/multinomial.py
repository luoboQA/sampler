import numpy as np
from scipy.special import gammaln  # 用于计算 log 阶乘，避免数值溢出


class MultinomialDistribution(object):
    """多项分布类，用于表示和操作多项分布随机变量"""

    def __init__(self, p, rso=np.random):
        """初始化多项分布

        Parameters
        ----------
        p : numpy array, shape (k,)
            每个类别的概率，总和必须为 1
        rso : numpy RandomState 对象，可选
            随机数生成器，默认为 np.random
        """
        # 检查概率之和是否为 1（允许浮点误差）
        if not np.isclose(np.sum(p), 1.0):
            raise ValueError("outcome probabilities do not sum to 1")

        self.p = p          # 保存概率向量
        self.rso = rso      # 保存随机数生成器
        # 预计算对数概率，用于 log-PMF 的高效计算
        self.logp = np.log(self.p)

    def sample(self, n):
        """从多项分布中采样 n 次试验的结果

        Parameters
        ----------
        n : int
            试验总次数

        Returns
        -------
        numpy array, shape (k,)
            每个类别出现的次数
            
        假设有6个面，概率均等
        p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        dist = MultinomialDistribution(p)

        掷10次骰子
        result = dist.sample(10)  
        可能输出: [2, 1, 3, 1, 2, 1]
        含义: 1点出现2次, 2点1次, 3点3次...
        """
        # 使用 NumPy 的多项分布采样函数
        x = self.rso.multinomial(n, self.p)
        return x

    def log_pmf(self, x):
        """计算给定样本 x 的对数概率质量函数（log-PMF）

        Parameters
        ----------
        x : numpy array, shape (k,)
            每个类别出现的次数

        Returns
        -------
        float
            log(P(X = x))
        """
        n = np.sum(x)                       # 总试验次数
        log_n_factorial = gammaln(n + 1)    # log(n!)
        sum_log_xi_factorial = np.sum(gammaln(x + 1))  # log(x1! * ... * xk!)

        # 计算 log(p1^x1 * ... * pk^xk)
        log_pi_xi = self.logp * x # = xi * log(pi)
        # 如果概率为 0 且 x[i] 也为 0，则乘积为 0，否则会得到 nan
        log_pi_xi[x == 0] = 0 # 避免 log(0) 产生 -inf,x == 0：返回一个布尔数组，标记哪些位置计数为0,把这些位置的值设为0,self.logp * x 会计算 0 * (-inf)得到 nan。手动将其设为 0 修正

        sum_log_pi_xi = np.sum(log_pi_xi)

        # 组合各部分得到 log-PMF
        log_pmf = log_n_factorial - sum_log_xi_factorial + sum_log_pi_xi
        return log_pmf

    def pmf(self, x):
        """计算概率质量函数（PMF）"""
        pmf = np.exp(self.log_pmf(x))
        return pmf
