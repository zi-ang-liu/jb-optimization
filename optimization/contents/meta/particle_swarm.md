# Particle Swarm Optimization

```{prf:algorithm} PSO
:label: pso

**Inputs:** population size $N$, inertia weight $w$, acceleration coefficients $c_1, c_2$   
**Output:** best individual $x_{gbest}$

1. $D \leftarrow$ dimension of $x$ 
2. Initialize population $P$ with $N$ individuals
3. **while** not converged **do**






```


% \subsection{Particle Swarm Optimization}

% \begin{algorithm}[H]
%   \caption{Particle Swarm Optimization}\label{alg:PSO}
%   \KwIn{population size $N$, inertia weight $w$, acceleration coefficients $c_1, c_2$}
%   \KwOut{best individual $x_{gbest}$}
%   $D \leftarrow$ dimension of $x$\;
%   Initialize population $P$ with $N$ individuals\;
%   \While{\textnormal{\texttt{not converged}}}{
%     \ForEach{$x \in P$}{
%       $r_1, r_2 \leftarrow$ random vectors in $[0, 1]^D$\;
%       $v \leftarrow wv + c_1r_1 \odot (x_{pbest} - x) + c_2r_2 \odot (x_{gbest} - x)$\;
%       $x \leftarrow x + v$\;
%       \If{$f(x) < f(x_{pbest})$}{
%         $x_{pbest} \leftarrow x$\;
%         \If{$f(x) < f(x_{gbest})$}{
%           $x_{gbest} \leftarrow x$\;
%         }
%       }
%     }
%   }
%   \Return{$x_{best}$}\;
% \end{algorithm}