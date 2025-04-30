import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.clust_dr import DistR
from src.affinities import SymmetricEntropicAffinity, NormalizedGaussianAndStudentAffinity, NormalizedLorentzHyperbolicAndStudentAffinity, EntropicAffinity
from src.utils_hyperbolic import plotPoincareFromLorentz
from src.utils_hyperbolic import lorentz_to_poincare


# Random seed
seed = 0
g = torch.Generator()
g.manual_seed(seed)

# Color maps used in the figure
cm_d = plt.cm.get_cmap('PuBuGn')
cm_e = plt.cm.get_cmap('YlOrBr')
cm_c = plt.cm.get_cmap('Purples')

# used for 3D plots
def _format_axes(ax):
    """Visualization options for the 3D axes."""
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    

# === Euclidean Embedding ===

G = nx.cycle_graph(20)

# 3d spring layout
pos = nx.spring_layout(G, dim=3, seed=779)
# Extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in sorted(G)])

X = torch.Tensor(node_xyz).double()

affinity_data = SymmetricEntropicAffinity(perp=5, lr=1e-2, max_iter=3000)
affinity_embedding = NormalizedGaussianAndStudentAffinity(student=False, sigma=1)#student=True)

model = DistR(affinity_data,affinity_embedding,
             init='normal',
             loss_fun='kl_loss',
             output_sam=10,
             lr=1e-1, 
             verbose=False)
# Fit and retrieve embedding
Z = model.fit_transform(X)

A = model.affinity_data.compute_affinity(X)
B = model.affinity_embedding.compute_affinity(Z)

B_perm = model.T @ B @ model.T.T
A_alpha = A.float().numpy()
A_alpha = A_alpha / A_alpha.max()
B_alpha = B.float().numpy()
B_alpha = B_alpha / B_alpha.max()


# === Hyperbolic Embedding ===

seed = 10

def make_blobs_random(n_samples, means, scale=1., seed=0):
  np.random.seed(seed) 
  per_blob=int(n_samples/(means.shape[0] + 5))
  result = np.random.randn(per_blob*2,3) * scale + means.mean(0)
  labels = np.zeros(per_blob*2)
  for r in range(1,means.shape[0]+1):
    per_blob_ = per_blob*2 if r%2==1 else per_blob
    new_blob = np.random.randn(per_blob_,3) * scale + means[r-1]
    result = np.vstack((result,new_blob))
    labels = np.hstack([labels,[r]*per_blob_])
  return result, labels

G_ = nx.cycle_graph(8)
# 3d spring layout
seed_layout = 2
pos_ = nx.spring_layout(G_, dim=3, seed=seed_layout)
means = np.array([pos_[v] for v in sorted(G_)])

n_samples = 42
samples, y = make_blobs_random(n_samples, means, scale=.1, seed=seed)
X_H = torch.from_numpy(samples).double()

affinity_embedding = NormalizedLorentzHyperbolicAndStudentAffinity(student=False, sigma=1.)
affinity_data = EntropicAffinity(perp=20)

n_points = 12
model_H = DistR(affinity_data=affinity_data, 
             affinity_embedding=affinity_embedding, 
             output_sam=n_points,  
             init_T="spectral", 
             lr=1e-2, 
             optimizer='RAdam', 
             loss_fun="kl_loss", 
             output_dim=2,
             init='WrappedNormal',
             seed=seed, 
             max_iter=1000, 
             max_iter_outer=10,
             verbose=False)

Z_H = model_H.fit_transform(X_H)

A_H = model_H.PX
B_H = model_H.affinity_embedding.compute_affinity(Z_H)

A_alpha_H = A_H.float().numpy()
A_alpha_H = A_alpha_H / A_alpha_H.max()
B_alpha_H = B_H.float().numpy()
B_alpha_H = B_alpha_H / B_alpha_H.max()


# === Plotting ===

params = {'text.usetex': False}
plt.rcParams.update(params)
plt.rc('font', family='DejaVu Serif')
ftsize=20


# Create the 3D figure
fig = plt.figure(figsize=(11,8))

ax = fig.add_subplot(221,  projection="3d")
# Plot the nodes
c_ = np.arange(node_xyz.shape[0])
ax.scatter(*node_xyz.T, s=100, alpha=1, ec="w", c=c_)

# Plot the edges
for i,vs in enumerate(node_xyz):
    for j in range(i,node_xyz.shape[0]):
        ax.plot([node_xyz[i,0],node_xyz[j,0]], 
                [node_xyz[i,1],node_xyz[j,1]],
                [node_xyz[i,2],node_xyz[j,2]],
                color=cm_d(A_alpha[i,j]),
                alpha=A_alpha[i,j])
_format_axes(ax)
ax.set_title(r'Input $\mathbf{X}$',fontsize=ftsize)

ax = fig.add_subplot(222)
# Plot the edges
for i  in range(Z.shape[0]):
    for j in range(i,Z.shape[0]):
        ax.plot([Z[i,0],Z[j,0]], 
                [Z[i,1],Z[j,1]],
                color=cm_e(B_alpha[i,j]),
                alpha=B_alpha[i,j], 
                linewidth=3,
                zorder=0)
ax.scatter(Z[:,0],Z[:,1], s=200 ,alpha=1, zorder=1, c=model.T.T @ np.array(c_))
# ax.set_xticks([])
# ax.set_yticks([])
ax.yaxis.tick_right()
ax.set_yticklabels([])
ax.set_xticklabels([])

ax.set_xlim(Z[:,0].min()-1, Z[:,0].max()+1)
ax.set_ylim(Z[:,1].min()-1, Z[:,1].max()+1)
ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title(r'Embedding $\mathbf{Z}$',fontsize=ftsize)
## --- hyperbolic part

ax = fig.add_subplot(223, projection="3d")
# Plot the nodes
ax.scatter(*samples.T, s=50, alpha=1, c=y, ec="k",cmap='tab10')
# Plot the edges
for i in range(samples.shape[0]):
    for j in range(i,samples.shape[0]):
        ax.plot([samples[i,0],samples[j,0]], 
                [samples[i,1],samples[j,1]],
                [samples[i,2],samples[j,2]],
                color=cm_d(A_alpha_H[i,j]),
                alpha=A_alpha_H[i,j])
_format_axes(ax)
ax.set_title(r'Input $\mathbf{X}$',fontsize=ftsize)

ax = fig.add_subplot(224)
plotPoincareFromLorentz(Z_H,model_H.T, torch.from_numpy(y).to(torch.int64), ax,lw=0.1,size_factor=60, thres=1)
#ax.set_title('Hyperbolic Embedding space',fontsize=20)
proj_poincare = lorentz_to_poincare(Z_H)

# Plot the edges
for i  in range(proj_poincare.shape[0]):
    for j in range(i,proj_poincare.shape[0]):
        ax.plot([proj_poincare[i,0],proj_poincare[j,0]], 
                [proj_poincare[i,1],proj_poincare[j,1]],
                color=cm_e(B_alpha_H[i,j]),
                alpha=B_alpha_H[i,j],
                linewidth=2,
                zorder=0)
ax.set_title(r'Embedding $\mathbf{Z}$',fontsize=ftsize)

# Insets
scale=.22
scale_coupling = 0.95
scale_Zaff = 0.8

inset = fig.add_axes([.515, 0.55, 0.1, scale])
inset.matshow(model.T, cmap=cm_c, aspect='auto')
inset.set_yticks([])
inset.set_xticks([])
inset.set_title(r'Coupling $\mathbf{T}$', fontsize=ftsize, y=-0.3)

inset = fig.add_axes([.515, 0.53, 0.1, 0.01])
inset.matshow(model.T.sum(0).reshape(1,-1),cmap=cm_c,vmin=0,vmax=1,  aspect='auto')
inset.set_yticks([])
inset.set_xticks([])

inset = fig.add_axes([.623, 0.55, 0.01, scale])
inset.matshow(model.T.sum(1).reshape(1,-1),cmap=cm_c,vmin=0,vmax=1,  aspect='auto')
inset.set_yticks([])
inset.set_xticks([])


inset = fig.add_axes([.315, .55, scale, scale])
inset.matshow(A, cmap=cm_d)
inset.set_yticks([])
inset.set_xticks([])
inset.set_title(r'$\mathbf{C}_X(\mathbf{X})$ (SEA)', fontsize=ftsize)

inset = fig.add_axes([.496, .78, 0.1375, 0.1375])
inset.matshow(B, cmap=cm_e)
inset.set_yticks([])
inset.set_xticks([])
inset.set_title(r'$\mathbf{C}_Z(\mathbf{Z})$ (Gaussian)', fontsize=ftsize)


inset = fig.add_axes([.515, 0.05, 0.1, scale])
inset.matshow(model_H.T, cmap=cm_c, aspect='auto')
inset.set_yticks([])
inset.set_xticks([])
inset.set_title(r'Coupling $\mathbf{T}$', fontsize=ftsize, y=-0.3)


inset = fig.add_axes([.515, 0.03, 0.1, 0.01])
inset.matshow(model_H.T.sum(0).reshape(1,-1),cmap=cm_c,vmin=0,vmax=1,  aspect='auto')
inset.set_yticks([])
inset.set_xticks([])

inset = fig.add_axes([.623, 0.05, 0.01, scale])
inset.matshow(model_H.T.sum(1).reshape(1,-1),cmap=cm_c,vmin=0,vmax=1,  aspect='auto')
inset.set_yticks([])
inset.set_xticks([])


inset = fig.add_axes([.315, .05, scale, scale])
inset.matshow(A_H, cmap=cm_d)
inset.set_yticks([])
inset.set_xticks([])
inset.set_title(r'$\mathbf{C}_X(\mathbf{X})$ (EA)', fontsize=ftsize)

inset = fig.add_axes([.496, .28, 0.1375, 0.1375])
inset.matshow(B_H, cmap=cm_e)
inset.set_yticks([])
inset.set_xticks([])
inset.set_title(r'$\mathbf{C}_Z(\mathbf{Z})$ (Lorentz)', fontsize=ftsize)

fig.tight_layout()
fig.subplots_adjust(wspace=.7)
# plt.text(.405, 0.22, r'Coupling $\mathbf{T}$', fontsize=ftsize)

plt.savefig('fig_general.pdf', bbox_inches='tight')
plt.show()

