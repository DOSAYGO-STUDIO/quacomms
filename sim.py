import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Bell state Phi+
bell00 = qt.tensor(qt.basis(2,0), qt.basis(2,0))
bell11 = qt.tensor(qt.basis(2,1), qt.basis(2,1))
psi0 = (bell00 + bell11).unit()

# Density matrix
rho = psi0 * psi0.dag()

# Operators
sigz_B = qt.tensor(qt.qeye(2), qt.sigmaz())

# Post-selection projectors for b=0 (Z basis)
proj_A0 = qt.tensor(qt.basis(2,0)*qt.basis(2,0).dag(), qt.qeye(2))
proj_A1 = qt.tensor(qt.basis(2,1)*qt.basis(2,1).dag(), qt.qeye(2))

# Conditional states for b=0
cond_rho_B0_0 = (proj_A0 * rho * proj_A0.dag()) / (proj_A0 * rho * proj_A0.dag()).tr()
cond_rho_B0_1 = (proj_A1 * rho * proj_A1.dag()) / (proj_A1 * rho * proj_A1.dag()).tr()
uncond_rho_B0 = 0.5 * cond_rho_B0_0 + 0.5 * cond_rho_B0_1
uncond_weak_val0 = qt.expect(sigz_B, uncond_rho_B0)

# For b=1 (X basis)
plus = (qt.basis(2,0) + qt.basis(2,1)).unit()
minus = (qt.basis(2,0) - qt.basis(2,1)).unit()
proj_Aplus = qt.tensor(plus * plus.dag(), qt.qeye(2))
proj_Aminus = qt.tensor(minus * minus.dag(), qt.qeye(2))

# Conditional states for b=1
cond_rho_B1_plus = (proj_Aplus * rho * proj_Aplus.dag()) / (proj_Aplus * rho * proj_Aplus.dag()).tr()
cond_rho_B1_minus = (proj_Aminus * rho * proj_Aminus.dag()) / (proj_Aminus * rho * proj_Aminus.dag()).tr()
uncond_rho_B1 = 0.5 * cond_rho_B1_plus + 0.5 * cond_rho_B1_minus
uncond_weak_val1 = qt.expect(sigz_B, uncond_rho_B1)

# Add stochastic noise (jitters) with bit-dependent variance bias
np.random.seed(42)
n_traj = 20
D_base = 0.1
epsilon0 = 0.05  # Small bias for b=0
epsilon1 = -0.05  # For b=1
jitters0 = uncond_weak_val0 + np.sqrt(2*(D_base + epsilon0)) * np.random.randn(n_traj)
jitters1 = uncond_weak_val1 + np.sqrt(2*(D_base + epsilon1)) * np.random.randn(n_traj)

# Means and variances
mean0, var0 = np.mean(jitters0), np.var(jitters0)
mean1, var1 = np.mean(jitters1), np.var(jitters1)

# Demon filtering: Filter high-variance subsets (e.g., abs > std)
thresh0 = np.std(jitters0)
filtered_jitters0 = jitters0[np.abs(jitters0 - mean0) > thresh0]
filtered_mean0 = np.mean(filtered_jitters0) if len(filtered_jitters0) > 0 else mean0

thresh1 = np.std(jitters1)
filtered_jitters1 = jitters1[np.abs(jitters1 - mean1) > thresh1]
filtered_mean1 = np.mean(filtered_jitters1) if len(filtered_jitters1) > 0 else mean1

# Output results
print(f"Unconditional weak value bit 0: {uncond_weak_val0}")
print(f"Unconditional weak value bit 1: {uncond_weak_val1}")
print(f"Mean jitter bit 0: {mean0}, Var: {var0}")
print(f"Mean jitter bit 1: {mean1}, Var: {var1}")
print(f"Filtered mean bit 0: {filtered_mean0}")
print(f"Filtered mean bit 1: {filtered_mean1}")

# Plot jitters
plt.figure()
plt.plot(range(n_traj), jitters0, label='Bit 0 jitters')
plt.plot(range(n_traj), jitters1, label='Bit 1 jitters')
plt.xlabel('Trajectory index')
plt.ylabel('Jittered weak value')
plt.legend()
plt.savefig('uncond_jitters_plot.png')
print("Plot saved as uncond_jitters_plot.png")
