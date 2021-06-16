# Multiplicative intrinsic component optimization (MICO)
## 1.背景介绍
图像分割是医学图像处理中的常见任务。在核磁共振成像(MRI)中，强度不均匀性是一种常见的固有伪影，它的存在给图像分割带来了很大的挑战。本文提出了一种解决强度不均匀性的算法，该算法通过估计图像的偏差场来实现MRI图像的偏差校正与组织的分割。
## 2.问题建模
MRI图像可以用如下公式来表示
$$
I(x) = b(x)J(x)+n(x)
$$
其中I(x)为图像在体素x处的强度，J(x)为真实图像，b(x)为偏差场，n(x)为噪声。图像I被分解成了两个相乘的内在成分b和J再额外加上噪声n。可以通过能量最小化问题来找到图像的乘法内在成分b和J，从而实现偏差校正与组织分割。即最小化如下函数 

$$
F(b,j) = \int_{\Omega} |I(x)-b(x)J(x)|^2{\rm d}x
$$

这是一个不适定问题，为了使问题可解，需要通过一些先验知识来限定b和J的搜索空间。对于真实图像J，假设组织之间是不相交的，则可以将J的搜索空间限定在由方程$\color{#62749C}J(x)=\sum_{i=1}^Nc_{i}u_{i}$构成的子空间，其中N为图像中人体组织的数量，u是二值函数，代表当前体素x是否有第i个组织，c为常数；对于偏差场b，假设偏差场在全局是平滑的且变化缓慢，在一个小的局部可以看作是不变的，则可以使用光滑基函数g的线性组合来表达偏差场$\color{#62749C}b(x)=\textbf{w}^TG(x)$，其中w为要求解的参数。
对b，J的搜索空间进行限定后，能量函数可以转化为

$$
F(b,j) = F(\textbf{u,c,w}) = \int_{\Omega}|I(x)-\textbf{w}^TG(x)\sum_{i=1}^Nc_{i}u_{i}(x)|^2{\rm d}x
$$

分别对u,c,w进行迭代优化，即可求解偏差场b和真实图像J。
## 3.实验结果
![result](data\result.png)

参考文献
[1]Chunming, Li, John, et al. Multiplicative intrinsic component optimization (MICO) for MRI bias field estimation and tissue segmentation[J]. Magnetic Resonance Imaging, 2014, 32(7):913-923.
