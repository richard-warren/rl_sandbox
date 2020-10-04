# iLQR ( in principle)



### overview
In **optimal control** we seek to control a system such that some cost function is minimized. [Some consider this an attractive model for biological motor control](https://homes.cs.washington.edu/~todorov/papers/TodorovNatNeurosci02.pdf) because it helps resolve difficulties associated with the redundancy of biomechanical systems. There are infinitely many ways to solve the same task (e.g. reaching to a target); by casting this as an optimization problem we can "simply" minimize some cost function (e.g. distance to target). This framework predicts that variability will persist along dimensions irrelevant to the cost function, a prediction [that has some experimental support](https://homes.cs.washington.edu/~todorov/papers/TodorovNatNeurosci02.pdf).

For simple systems with linear dynamics and quadratic cost functions there exist well-studied methods for finding (provably) optimal control strategies. [LQR](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) finds a feedback matrix $\mathbf{K}$ that linearly maps the state of the system to control inputs that are optimal with respect to a quadratic cost function (although this is only possible when the system is ["controllable!"](https://www.youtube.com/watch?v=u5Sv7YKAkt4)). Hope is not lost when the dynamics and cost function aren't linear and quadratic. We can use Taylor expansion to find first and second order approximations of the dynamics and cost, then apply LQR methods (albeit without the same theoretical guarantees).

[iLQR](https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf) is an iterative version of LQR (and is slightly simplified version of [Differential Dynamic Programming](https://en.wikipedia.org/wiki/Differential_dynamic_programming)). LQR finds a single feedback rule that linearly maps states to actions; this rule could be defined using an approximation of the dynamics around a single point. But if we stray too far from that point the approximation may fail. In iLQR we iteratively optimize control inputs for *entire trajectories*, defining feedback (and feed-forward) control laws for each time step in the trajectory. Regularization is used to ensure we don't stray too far from the points around which are approximations are made. iLQR can find really impressive behaviors for finite-horizon problems, and can be used for [online trajectory optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf) if controls are optimized at each timestep for some finite horizon.

### maths

Consider a discrete-time, finite horizon system with state $\mathbf{x} \in \mathbb{R}^n$. the system dynamics evolve according to the previous state and actions (control inputs):

$$
\begin{equation}
\mathbf{x}_{i+1} = \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i)
\end{equation}
$$

notice that all states depend only on the control sequence and the initial state - things unravel deterministically from there...

we want to maximize the 'goodness' (minimize the 'badness') of the trajectory. A trajectory is defined by a sequence of states ${ \mathbf{x}_0, \mathbf{x}_1, \dots \mathbf{x}_N}$ and actions $U \equiv { \mathbf{x}_0, \mathbf{x}_1, \dots \mathbf{x}_N}$.  'badness' is defined by a cost function of the form:

$$ J(\mathbf{x}, \mathbf{U}) = \sum_{i=0}^{N-1} \ell(\mathbf{x}_i, \mathbf{u}_1) + \ell_f(\mathbf{x}_N) $$

notice that there are two costs: the first varies with states and controls. this means we can penalize eg control effort and distance to target eg. the final cost only varies with final state eg distance to target. we can minimize control effort in first cost!!!

we want to find the U that minimizes J. to find the best u we rely on the principle of optimality. at each time we want to pick the actions that minize the immediate cost plus the sum of future costs. if we do that at every time, we have an optimal trajectory! we will (confusingly) call the value of a state at time i (notice that the value of a state depends on the time!!!) the cost we expect assuming we act optimally from there on out:

$$
\begin{align}
V(\mathbf{x},i)
&= \min_{\mathbf{U}} J_i(\mathbf{x}, \mathbf{U}) \nonumber \\
&= \min_{\mathbf{u}} [\ell(\mathbf{x}, \mathbf{u}) + V(\mathbf{x_{i+1}},i+1)] \nonumber \\
&= \min_{\mathbf{u}} [\ell(\mathbf{x}, \mathbf{u}) + V(\mathbf{f}(\mathbf{x}, \mathbf{u}),i+1)]
\end{align}
$$

remember that the value is only a function of the state (because we assumes optimal actions). this means that given current state x, we just need to pick the current u, which together determine both the immediate loss and the value at the next state. we can reason about the optimality of trajectories by sequentially reasoning about the optimality of individual steps!

so far so good, but how do we pick at optimal u at a single time step. imagine we started with an intial action sequence U, which determined (via the initial state and the dynamics function) a sequence of states. now, positioning ourselves at some time i within that trajectory, we are going to build a quadratic approximation of how the argument to $(2)$ differs from our current xi ui:

$$
\begin{align}
Q(\mathbf{x}, \mathbf{u})

&= \ell(\mathbf{x},\mathbf{u}) + V'(\mathbf{f}(\mathbf{x},\mathbf{u})) - [\ell(\mathbf{x}_i,\mathbf{u}_i) + V'(\mathbf{f}(\mathbf{x}_i,\mathbf{u}_i))] \\

&\approx Q(\mathbf{x}_i,\mathbf{u}_i) + Q_{\mathbf{x}}(\mathbf{x}_i,\mathbf{u}_i)^\intercal (\mathbf{x}-\mathbf{x}_i) + Q_{\mathbf{u}}(\mathbf{x}_i,\mathbf{u}_i)^\intercal(\mathbf{u}-\mathbf{u}_i) + \frac{1}{2} (\mathbf{x}-\mathbf{x}_i)^\intercal Q_{\mathbf{xx}}(\mathbf{x}_i,\mathbf{u}_i)(\mathbf{x}-\mathbf{x}_i) + (\mathbf{u}-\mathbf{u}_i)^\intercal Q_{\mathbf{ux}}(\mathbf{x}_i,\mathbf{u}_i)(\mathbf{x}-\mathbf{x}_i) + (\mathbf{u}-\mathbf{u}_i)^\intercal Q_{\mathbf{uu}}(\mathbf{x}_i,\mathbf{u}_i)(\mathbf{u}-\mathbf{u}_i)
\end{align}
$$

that is awful. we'll clean up the notation by setting $\mathbf{\delta u}=(\mathbf{u}-\mathbf{u}_i)$, $\mathbf{\delta x}=(\mathbf{x}-\mathbf{x}_i)$, dropping the $(\mathbf{x}_i,\mathbf{u}_i)$ dependencies, and noticing that $Q(\mathbf{x}_i, \mathbf{u}_i)=0$. now our function describes how cost to go changes as we nudge x and u around.

$$
\begin{align}
Q(\mathbf{\delta x}, \mathbf{\delta u})
&\approx Q_{\mathbf{x}}^\intercal \mathbf{\delta x} + Q_{\mathbf{u}}^\intercal \mathbf{\delta u} + \frac{1}{2} \mathbf{\delta x}^\intercal Q_{\mathbf{xx}}\mathbf{\delta x} + \mathbf{\delta u}^\intercal Q_{\mathbf{ux}}\mathbf{\delta x} + \frac{1}{2}\mathbf{\delta u}^\intercal Q_{\mathbf{uu}}\mathbf{\delta u}
\end{align}
$$

differentiating wrt du (all Qs are constant!) and setting to 0:

$$
\begin{align}
\frac{\mathbf{\delta Q}}{\mathbf{\delta u}} &= 0 \\
Q_{\mathbf{u}} + Q_{\mathbf{ux}}\mathbf{\delta x} + Q_{\mathbf{uu}}\mathbf{\delta u} &= 0 \\
\mathbf{\delta u} &= -Q_{\mathbf{uu}}^{-1}(Q_{\mathbf{u}} + Q_{\mathbf{ux}}\mathbf{\delta x})
\end{align}
$$

feedforward and feedback terms k and K...
$$
\begin{align}
\mathbf{k} &= -Q_{\mathbf{uu}}^{-1} Q_{\mathbf{u}} \\
\mathbf{K} &= -Q_{\mathbf{uu}}^{-1} Q_{\mathbf{ux}}
\end{align}
$$

to update our actions we add $\mathbf{\delta u}$ to $\mathbf{u}. notice dx is the difference between our new and old trajectory...

$$ \hat{\mathbf{u}} = \mathbf{u} + \mathbf{k} + \mathbf{K \delta x} $$

chain rule gives expansion coeffs:

$$
\begin{align}
Q_{\mathbf{x}}  &= \ell_{\mathbf{x}} + \mathbf{f}_{\mathbf{x}}^\intercal V'_{\mathbf{x}} \\

Q_{\mathbf{u}}  &= \ell_{\mathbf{u}} + \mathbf{f}_{\mathbf{u}}^\intercal V'_{\mathbf{x}}\\

Q_{\mathbf{xx}} &= \ell_{\mathbf{xx}} + \mathbf{f}_{\mathbf{x}}^\intercal V'_{\mathbf{xx}}\mathbf{f}_{\mathbf{x}} + V'_{\mathbf{x}}\mathbf{f}_{\mathbf{xx}} \\

Q_{\mathbf{uu}} &= \ell_{\mathbf{uu}} + \mathbf{f}_{\mathbf{u}}^\intercal V'_{\mathbf{xx}}\mathbf{f}_{\mathbf{u}} + V'_{\mathbf{x}}\mathbf{f}_{\mathbf{uu}} \\

Q_{\mathbf{ux}} &= \ell_{\mathbf{xu}} + \mathbf{f}_{\mathbf{u}}^\intercal V'_{\mathbf{xx}}\mathbf{f}_{\mathbf{x}} + V'_{\mathbf{x}}\mathbf{f}_{\mathbf{ux}}

\end{align}
$$

notice that depends on V for next state! getting this state requires knowing next state, meaning we work backwords from the final state, for which derivs are easy. how do we find V terms? plugging bla into bloom we have:

$$
\begin{align}
\bigtriangleup V &= -\frac{1}{2} Q_\mathbf{u}^\intercal Q_\mathbf{uu}^{-1}Q_\mathbf{u} \\
V_\mathbf{x} &= Q_\mathbf{x} - \frac{1}{2} Q_\mathbf{u}^\intercal Q_\mathbf{uu}^{-1}Q_\mathbf{ux} \\
V_\mathbf{xx} &= Q_\mathbf{xx} - \frac{1}{2} Q_\mathbf{xu}^\intercal Q_\mathbf{uu}^{-1}Q_\mathbf{ux}
\end{align}
$$

notice depends on dx. as we nudge our actions around our trajectory will deviate from the previous one. feedback will act on these deviations, but if we change things too aggressively we will out of the range where our approximation is accurate...

notice that delta v tells us how much reduction to expect, assuming our quadratic approximation is accurate. this is nice bc...

### regularization
but wait! what if we can't invert! regularize by adding off diagonal terms. amounts to quadratic costs on control deviations. when the algo is going well (cost decreasing) we can reduce regulization and act more aggresively. when going poorly (cost increasing OR non-pd (invertable???)) we increase it. explain how this trades off between gauss newton and gradient descent.

there's a nice interpretation of this reg method // in gradient descent we move in the (opposite) direciton of the gradient, which is akin to moving in the direction of a first order approximation // gauss newton uses quadratic approx // by taking into account second deriv we can converge faster




### algorithm

so where does that leave us? we start with some sequence of actions, and we want to find tweaks to that sequence that are optimal wrt our approximation of the cost function. computing those tweaks requires .... so we
1. compute derivates along trajectory along with states
2. work backwords, at each time compute ... using previous derivs and V estimates from previous timestep, k and K
3. apply k and K to generate new trajcetory. keep or lose and modify reg as necessary
