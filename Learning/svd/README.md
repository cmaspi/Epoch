# Singular Value Decomposition
Singular Value Decomposition is a mathematical trick to break any arbitrary matrix matrix which would have some nice properties.

A good way to introduce SVD is by asking "How does $\left|\left|{x}\right|\right|$ compare to $\left|\left|{Ax}\right|\right|$

Given any matrix $A$, consider the gram matrix $A^TA$, where the matrix $A$ has the dimensions $n \times m$. The gram matrix would have the dimensions $m \times m$

Using Spectral Value Decomposition, we can break the gram matrix as

$$ A^TA = VDV^T$$

where $D$ is a diagonal matrix, and $V$ is an orthogonal matrix

$$
\begin{align}
&V: m \times m\ \text{orthogonal}\\
&D: m \times m\ \text{diagonal}
\end{align}
$$
