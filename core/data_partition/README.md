#! https://zhuanlan.zhihu.com/p/373077728
### 基于Equiquantization的数据离散化

#### 参考文献

本文提供了一种基于数据信息量计算其对目标预测能力的方式: 

Georges A. Darbellay: Predictability: An Information-Theoretic Perspective, Signal Analysis and Prediction, 1998.

---

#### 算法理论

**1. 信息和可预测性**

数据预测取决于数据之间的依赖关系(dependence), 比如当我们尝试进行预测时, 我们会假设未来是依赖过去的. 如果数据之间不存在依赖, 也就无法执行预测.

随机变量$\bm{X}$和$\bm{Y}$随机独立, 当且仅当:

$$
\begin{aligned}
p_{x,y}(\bm{x}, \bm{y}) = p_x(\bm{x})p_y(\bm{y}), \forall \bm{x} \in \mathbb{R}^n, \forall \bm{y} \in \mathbb{R}^m
\end{aligned}
$$

当二者不存在依赖关系时:

$$
\begin{aligned}
\ln \frac{p_{x,y}(\bm x, \bm y)}{p_x(\bm x)p_y(\bm y)}
\end{aligned}
$$

取期望有：

$$
\begin{aligned}
I(\bm X; \bm Y) = \int_{\mathbb R^{n + m}} p_{x,y}(\bm x, \bm y) \ln \frac{p_{x,y}(\bm x, \bm y)}{p_x(\bm x)p_y(\bm y)} \mathrm{d}\bm{x} \mathrm{d}\bm{y}
\end{aligned}
$$

上式即为连续随机变量$\bm{X}$和$\bm{Y}$之间的互信息熵定义. 其中:

$$
\left\{
    \begin{aligned}
    &I(\bm X; \bm Y) = 0, \text{if } \bm X \perp \bm Y;\\
    &I(\bm X; \bm Y) > 0, \text{else }.
    \end{aligned}
\right.
$$

由于$I(\bm X; \bm Y)$无上界, 文献中为了对可预测性进行标准量化, 定义了如下的指标:

$$
\begin{aligned}
\rho = \sqrt{1 - e^{-2I(\bm X, \bm Y)}} 
\end{aligned}
$$

易有:

$$
\left\{
    \begin{aligned}
    &\rho(\bm X, \bm Y) = 0, \bm Y \text{ cannot be predicted by } \bm X;\\
    &\rho(\bm X, \bm Y) = 1, \text{this is the limit of determinism}; 
    \end{aligned}
\right.
$$

一般对于$\forall \bm U=f(\bm X)$, 其中$f$为一函数, 那么有对于$\bm Y$的预测能力:

$$
\begin{aligned}
\rho(\bm U, \bm Y) \leq \rho(\bm X, \bm Y)
\end{aligned}
$$

**2. 线性可预测度**

当$\bm X$和$\bm Y$之间存在线性关系时, 考虑到变量值测量时存在**高斯噪声**, 令$\bm Z = (\bm X, \bm Y)$, 那么:

$$
\begin{aligned}
I(\bm X; \bm Y) = \frac{1}{2} \ln \frac{\det \sum_{xx}\det\sum_{yy}}{\det \sum}
\end{aligned}
$$

其中$\sum_{xx}$、$\sum_{yy}$, $\sum$分别为$\bm X$, $\bm Y$和$\bm Z$的协方差矩阵. 由根据线性相关系数定义:

$$
\begin{aligned}
r_{ij} = r(X_i, Y_j) = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}}
\end{aligned}
$$

当$\bm X$和$\bm Y$均为1维时:

$$
\begin{aligned}
I(X;Y) = -\frac{1}{2}\ln(1-r^2(X,Y))
\end{aligned}
$$

那么$\rho(X,Y) = |r(X,Y)|$, 从而有:

$$
\begin{aligned}
\rho^2(\bm X, Y) = 1 - \frac{\det \sum}{\sigma_{yy} \det \sum_{xx}}
\end{aligned}
$$

根据Schur完备公式, 进一步将上式化为:

$$
\begin{aligned}
\rho^2(\bm X, Y) = \frac{\bm\sigma_{xy}^{tr} \sum_{xx}^{-1} \bm\sigma_{xy}}{\sigma_{yy}}
\end{aligned}
$$

更一般地:

$$
\begin{aligned}
\rho^2(\bm X, \bm Y) = 1 - \frac{\det\sum}{\det\sum_{xx}\det\sum_{yy}}
\end{aligned}
$$

定义线性可预测度为:

$$
\begin{aligned}
\lambda(\bm X, \bm Y) = \sqrt{1 - \frac{\det\sum}{\det\sum_{xx}\det\sum_{yy}}}
\end{aligned}
$$

**3. 对建模输入的选择**

设$\bm X_{n-1} = (X_1, ..., X_{n-1})$, $\bm X_{n} = (X_1, ..., X_{n})$, 根据互信息的链式法则有:

$$
\begin{aligned}
I(\bm X_n, \bm Y) = I(\bm X_{n-1}, \bm Y) + I(X_n, \bm Y|\bm X_{n-1})
\end{aligned}
$$

那么:

$$
\begin{aligned}
\rho(\bm X_{n-1}, \bm Y) \leq \rho(\bm X_{n}, \bm Y) 
\end{aligned}
$$

因为互信息总是非负数的, 那么:

$$
\begin{aligned}
\rho^2(X_n, \bm Y|\bm X_{n-1}) = \frac{\rho^2(\bm X_{n}, Y)-\rho^2(\bm X_{n-1}, Y)}{1-\rho^2(\bm X_{n-1}, Y)} \geq 0 
\end{aligned}
$$

不断选入$X$样本, $n$也不断增加, 当上式的值为0时, 说明$X_n$不再提供$\bm X_{n-1}$关于$Y$的新的信息. 那么可以使用$\rho(X_n, \bm Y|\bm X_{n-1})$来对建模输入进行选择.

---

#### 基于样本的信息熵估计

**1. marginal equiquantization分箱**

对于由$X$、$Y$构成的二维样本, 每一步分裂时都分别从$X$或$Y$维度上将样本按照样本数等分为两份, 这样原来的一个区间就被分为了四个概率相等的子区间. 使用$C_k$代表划分的第$k$个区间.

根据前面的互信息积分公式有:

$$
\begin{aligned}
I(\bm X; \bm Y) &= \sum_k \int_{C_k} \mathrm{d}\bm x\mathrm{d}\bm y p_{x,y}(\bm x, \bm y) \ln \frac{p_{x,y}(\bm x, \bm y)}{p_{x}(\bm x)p_{y}(\bm y)} \\
&= \sum_k \int_{C_k} \mathrm{d}\bm x\mathrm{d}\bm y p_{x,y}(\bm x, \bm y) \ln \frac{p_{x,y}(\bm x, \bm y)}{c_k p_{x}(\bm x)p_{y}(\bm y)} + \sum_k P_{\bm X, \bm Y}(C_k) \ln \frac{P_{\bm X,\bm Y}(C_k)}{P_{\bm X}(C_k)P_{\bm Y}(C_k)}
\end{aligned}
$$

其中:

$$
\begin{aligned}
c_k = \frac{P_{\bm X, \bm Y}(C_k)}{P_{\bm X}(C_k)P_{\bm Y}(C_k)} 
\end{aligned}
$$

$$
\begin{aligned}
P_{\bm X, \bm Y} = \int_{C_k} \mathrm{d}\bm x\mathrm{d}\bm y p_{x,y}(\bm x, \bm y) 
\end{aligned}
$$

本文证明, 当分箱区间$\Delta_1, ..., \Delta_d \rightarrow 0$时, 

$$
\begin{aligned}
\mathop{\lim}\limits_{\Delta_1, ..., \Delta_d \rightarrow 0} c_k = \frac{p_{x,y}(\bm x, \bm y)}{p_x(\bm x)p_y(\bm y)} 
\end{aligned}
$$

那么上式等号右侧第一项为0, 那么

$$
\begin{aligned}
I(\bm X; \bm Y) =  \sum_k P_{\bm X, \bm Y}(C_k) \ln \frac{P_{\bm X,\bm Y}(C_k)}{P_{\bm X}(C_k)P_{\bm Y}(C_k)} 
\end{aligned}
$$

前提条件是, 对于$C_k$的所有子集$C_{kl}$:

$$
\begin{aligned}
\frac{P_{\bm X,\bm Y}(C_{kl})}{P_{\bm X}(C_{kl})P_{\bm Y}(C_{kl})} = \frac{P_{\bm X,\bm Y}(C_k)}{P_{\bm X}(C_k)P_{\bm Y}(C_k)}, \forall l
\end{aligned}
$$

这一步可以通过严格统计检验来实现, 但为了算法方便, 我采用了一个$\varepsilon$作为收敛判据. 最后的分箱效果如下:

![Image](https://pic4.zhimg.com/80/v2-cbf96d64a3220c5acc67fe4ef7399e18.png)

**2. 信息熵估计**

通过下式对信息熵进行估计:

$$
\begin{aligned}
\hat{I}(\bm X, \bm Y) = \frac{1}{N} \sum_{k=1}^{m} N_{\bm X, \bm Y}(C_k) \ln \frac{N_{\bm X, \bm Y}(C_k)}{N_{\bm X}(C_k)N_{\bm Y}(C_k)} + \ln N
\end{aligned}
$$

其中$N_{\bm X, \bm Y}(C_k)$为在箱子$k$中的样本数, 而$N_{\bm X}(C_k)$和$N_{\bm Y}(C_k)$为总体样本$N$中分别处于$C_k$在$\bm X$和$\bm Y$方向投影区间内的样本数.

---

#### 代码地址

* [数据离散化](https://github.com/Ulti-Dreisteine/data-information-measurement/tree/main/core/data_partition)
* [互信息计算](https://github.com/Ulti-Dreisteine/data-information-measurement/blob/main/core/entropy/binning_based/marginal_equiquant.py)

