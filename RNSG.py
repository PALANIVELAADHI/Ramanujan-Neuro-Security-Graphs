import numpy as np
import networkx as nx
from scipy.special import gcd  # for gcd in Ramanujan sum
import matplotlib.pyplot as plt
from typing import Tuple, List

# ────────────────────────────────────────────────
# 1. Ramanujan Sum c_q(n)  (core entropy source)
# ────────────────────────────────────────────────
def ramanujan_sum(q: int, n: int) -> complex:
    """
    Compute Ramanujan's sum c_q(n) = sum_{k=1..q, gcd(k,q)=1} exp(2π i k n / q)
    Returns real part (imaginary should be ~0 due to symmetry)
    """
    if q == 1:
        return 1.0
    s = 0.0 + 0j
    for k in range(1, q + 1):
        if gcd(k, q) == 1:
            s += np.exp(2j * np.pi * k * n / q)
    return s.real  # imaginary part ≈ 0


def adaptive_entropy(q: int, n: int, alpha: float = 1.0, M: int = 256) -> int:
    """
    ξ_t = α * c_q(n) mod M   (scaled and modularized for integer state)
    """
    c = ramanujan_sum(q, n)
    val = int(round(alpha * c)) % M
    return val


# ────────────────────────────────────────────────
# 2. Ramanujan-like graph (using good expander proxy)
# ────────────────────────────────────────────────
def create_ramanujan_proxy_graph(n_nodes: int, degree: int = 4, seed: int = 42) -> nx.Graph:
    """
    Use NetworkX random regular expander as proxy for Ramanujan graph.
    For true Ramanujan, one would use LPS or explicit constructions (hard for arbitrary n).
    """
    np.random.seed(seed)
    G = nx.random_regular_expander_graph(n_nodes, degree, epsilon=0.1, seed=seed)
    print(f"Created expander graph: {n_nodes} nodes, degree {degree}, "
          f"spectral gap estimate: {nx.spectral_gap(G):.4f}")
    return G


# ────────────────────────────────────────────────
# 3. RNSG Core Class
# ────────────────────────────────────────────────
class RNSG:
    def __init__(self,
                 n_nodes: int = 64,          # |V|
                 degree: int = 4,
                 state_range: Tuple[int, int] = (0, 255),  # 8-bit states
                 activation: str = 'tanh',   # or 'relu', 'sigmoid', etc.
                 seed: int = 42):
        
        self.n = n_nodes
        self.d = degree
        self.min_state, self.max_state = state_range
        self.G = create_ramanujan_proxy_graph(n_nodes, degree, seed)
        self.A = nx.adjacency_matrix(self.G).toarray().astype(float)  # adjacency matrix
        
        # Initial security state K_0 ∈ [min,max]^n
        np.random.seed(seed)
        self.K = np.random.randint(self.min_state, self.max_state + 1, size=n_nodes)
        
        # Activation function
        if activation == 'tanh':
            self.f = np.tanh
        elif activation == 'relu':
            self.f = lambda x: np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation")
        
        # History for non-Markovian check
        self.history = []
    
    def update(self, t: int,
               q_base: int = 17, alpha: float = 1.5,
               adapt_freq: int = 5, delta_d_max: int = 1) -> np.ndarray:
        """
        One evolution step: K_{t+1} = f( A K_t + ξ_t )
        With adaptive entropy and occasional edge adaptation
        """
        # Adaptive modulus for entropy (slowly changing)
        q = q_base + (t // 10) % 30   # example: cycles through larger q
        
        # Generate entropy vector ξ_t (one scalar per node, but correlated via Ramanujan sum)
        n_for_entropy = t + 1
        xi_scalar = adaptive_entropy(q=q, n=n_for_entropy, alpha=alpha, M=32)
        xi = np.full(self.n, xi_scalar) + np.random.randint(-2, 3, self.n)  # small noise
        
        # Linear diffusion via graph + entropy
        diffusion = self.A @ self.K.astype(float)
        perturbed = diffusion + xi
        
        # Nonlinear activation + scale back to integer range
        new_K_float = self.f(perturbed / self.d) * (self.max_state - self.min_state) + \
                      (self.min_state + self.max_state) / 2
        self.K = np.clip(np.round(new_K_float), self.min_state, self.max_state).astype(int)
        
        # Occasional adaptation of graph (edge re-weighting / small rewiring)
        if t % adapt_freq == 0 and t > 0:
            self._adapt_graph(delta_d_max)
        
        self.history.append(self.K.copy())
        return self.K
    
    def _adapt_graph(self, delta_d_max: int):
        """Simple adaptation: randomly rewire small number of edges (preserve degree-ish)"""
        edges = list(self.G.edges())
        to_remove = np.random.choice(len(edges), size=min(delta_d_max * self.n // 10, len(edges)//4), replace=False)
        for idx in to_remove:
            u, v = edges[idx]
            self.G.remove_edge(u, v)
        
        # Add new random edges (try to preserve regularity)
        candidates = [(i,j) for i in range(self.n) for j in range(i+1,self.n) if not self.G.has_edge(i,j)]
        np.random.shuffle(candidates)
        for u,v in candidates[:len(to_remove)]:
            self.G.add_edge(u, v)
        
        # Recompute A (in real impl → incremental update!)
        self.A = nx.adjacency_matrix(self.G).toarray().astype(float)
    
    def entropy_estimate(self, window: int = 20) -> List[float]:
        """Rolling Shannon entropy of last window states (per node averaged)"""
        if len(self.history) < window:
            return []
        ents = []
        for i in range(len(self.history) - window + 1):
            chunk = np.array(self.history[i:i+window])
            # Simple empirical entropy per position
            ent = 0.0
            for col in chunk.T:
                p = np.bincount(col, minlength=self.max_state+1) / window
                p = p[p>0]
                ent += -np.sum(p * np.log2(p + 1e-10))
            ents.append(ent / self.n)
        return ents


# ────────────────────────────────────────────────
# Example Simulation & Plots
# ────────────────────────────────────────────────
if __name__ == "__main__":
    rns = RNSG(n_nodes=64, degree=4, state_range=(0,255), activation='tanh', seed=123)
    
    T = 150
    states_over_time = []
    entropies = []
    
    for t in range(T):
        K_new = rns.update(t, q_base=13, alpha=2.0, adapt_freq=8)
        states_over_time.append(K_new.mean())  # track average state (for visualization)
        if t >= 20:
            ent = rns.entropy_estimate(window=20)[-1]
            entropies.append(ent)
    
    # Plot 1: Average state evolution
    plt.figure(figsize=(10,4))
    plt.plot(states_over_time, label='Mean security state')
    plt.xlabel("Time step t")
    plt.ylabel("Average K_t value")
    plt.title("RNSG State Evolution (tanh activation)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot 2: Empirical entropy growth
    plt.figure(figsize=(10,4))
    plt.plot(range(20, T), entropies, color='orange', label='Rolling entropy (window=20)')
    plt.xlabel("Time step t")
    plt.ylabel("Average entropy per node (bits)")
    plt.title("Entropy Growth under Adaptive Ramanujan Injection")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print("\nFinal state sample (first 10 nodes):", rns.K[:10])
    print("Estimated final entropy:", entropies[-1] if entropies else "N/A")