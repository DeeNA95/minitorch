$Y = A \times B$, the backward gradient flowing into $B$ is calculated as: $A^T \times \frac{\partial L}{\partial Y}$.

From linear algebra, if $Y = W \times C_{col}$, the gradient for the weights $W$ is: $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \times C_{col}^T$
