### 基于Equiquantization的数据离散化

#### 参考文献

本文提供了一种基于数据信息量计算其对目标预测能力的方式.

> Georges A. Darbellay: Predictability: An Information-Theoretic Perspective, Signal Analysis and Prediction, 1998.

#### 算法理论

**1. 信息和可预测性**

数据预测取决于数据之间的依赖关系(dependence), 比如当我们尝试进行预测时, 我们会假设未来是依赖过去的. 如果数据之间不存在依赖, 也就无法执行预测.

随机变量$\bm{X}$和$\bm{Y}$随机独立, 当且仅当:

$$
\begin{aligned}
p_{x,y}(\bm{x}, \bm{y}) = p_x(\bm{x})p_y(\bm{y}), \forall \bm{x} \in \mathbb{R}^n, \forall \bm{y} \in \mathbb{R}^m \tag{1}
\end{aligned}
$$

当二者不存在依赖关系时:

$$
\begin{aligned}
\ln \frac{p_{x,y}(\bm x, \bm y)}{p_x(\bm x)p_y(\bm y)} \tag{2} 
\end{aligned}
$$

取期望有：

$$
\begin{aligned}
I(\bm X; \bm Y) = \int_{\mathbb R^{n + m}} p_{x,y}(\bm x, \bm y) \ln \frac{p_{x,y}(\bm x, \bm y)}{p_x(\bm x)p_y(\bm y)} \mathrm{d}\bm{x} \mathrm{d}\bm{y} \tag{3}
\end{aligned}
$$

上式即为连续随机变量$\bm{X}$和$\bm{Y}$之间的互信息熵定义. 其中:

$$
\left\{
    \begin{aligned}
    &I(\bm X; \bm Y) = 0, \text{if } \bm X \perp \bm Y; \\
    &I(\bm X; \bm Y) > 0, \text{else }.
    \end{aligned}
\right.
$$

由于$I(\bm X; \bm Y)$无上界, 文献中为了对可预测性进行标准量化, 定义了如下的指标:

$$
\begin{aligned}
\rho = \sqrt{1 - \exp(-2I)}
\end{aligned}
$$
