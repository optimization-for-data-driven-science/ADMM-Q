import numpy as np
from itertools import combinations 
import os
import matplotlib.pyplot as plt
import pickle


def obj_val(Q, b, x):

	return 0.5 * (x.T @ Q) @ x + b.T @ x

def my_rint(x, delta_=8.0):

	return np.rint(x / delta_) * delta_


v = 10
d = 16
D = range(5)

admm = np.array([])
admm_min = np.array([])
admm_max = np.array([])
admms = np.array([])
admms_min = np.array([])
admms_max = np.array([])
admmr = np.array([])
admmr_min = np.array([])
admmr_max = np.array([])
pgd = np.array([])
pgd_min = np.array([])
pgd_max = np.array([])

for i in range(12):
	script_dir = os.path.dirname(__file__)
	rel_path = "tmp/soft_%d_%d_%d.tmp" % (d, v, i)
	abs_file_path = os.path.join(script_dir, rel_path)
	with open(abs_file_path, "r") as f:
		tmp = f.readlines()

		admm_min = np.concatenate((admm_min, np.array([float(tmp[0].strip())])))
		admm = np.concatenate((admm, np.array([float(tmp[1].strip())])))
		admm_max = np.concatenate((admm_max, np.array([float(tmp[2].strip())])))

		admms_min = np.concatenate((admms_min, np.array([float(tmp[3].strip())])))
		admms = np.concatenate((admms, np.array([float(tmp[4].strip())])))
		admms_max = np.concatenate((admms_max, np.array([float(tmp[5].strip())])))


		admmr_min = np.concatenate((admmr_min, np.array([float(tmp[6].strip())])))
		admmr = np.concatenate((admmr, np.array([float(tmp[7].strip())])))
		admmr_max = np.concatenate((admmr_max, np.array([float(tmp[8].strip())])))


		pgd_min = np.concatenate((pgd_min, np.array([float(tmp[9].strip())])))
		pgd = np.concatenate((pgd, np.array([float(tmp[10].strip())])))
		pgd_max = np.concatenate((pgd_max, np.array([float(tmp[11].strip())])))

# Uncomment the following line to pick the 5 instances that has the closest median 

# idx = list(range(12))
# best_idx = None
# best_std = 10 ** 100
# for item in combinations(idx, 5):
# 	tmp = admm[list(item)]
# 	tmp_std = np.std(tmp)
# 	if tmp_std < best_std:
# 		best_std = tmp_std
# 		best_idx = list(item)

best_idx = np.arange(5)

admm = admm[best_idx]
admm_min = admm_min[best_idx]
admm_max = admm_max[best_idx]
admms = admms[best_idx]
admms_min = admms_min[best_idx]
admms_max = admms_max[best_idx]
admmr = admmr[best_idx]
admmr_min = admmr_min[best_idx]
admmr_max = admmr_max[best_idx]
pgd = pgd[best_idx]
pgd_min = pgd_min[best_idx]
pgd_max = pgd_max[best_idx]




x_axis = D
print(x_axis)
fig = plt.figure(1, figsize=(8.5, 5.5))

plt.rc('font', size=10)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({"font.size": 15})


ax = fig.add_subplot(2, 2, 1)
x_axis = D
ax.errorbar(x_axis, np.array(admm), xerr=None, yerr=[admm - admm_min, admm_max - admm], fmt="ro", capsize=6, label=r"$f(x_{\texttt{ADMM-Q}})$", linewidth=1.25, elinewidth=1.25)
ax.errorbar(x_axis, np.array(admms), xerr=None, yerr=[admms - admms_min, admms_max - admms], fmt="bo", capsize=6, label=r"$f(x_{\texttt{ADMM-S}})$", linewidth=1.25, elinewidth=1.25)
ax.legend(loc="upper right")
ax.set_xticks(x_axis)



ax = fig.add_subplot(2, 2, 2)
x_axis = D
ax.errorbar(x_axis, np.array(admm), xerr=None, yerr=[admm - admm_min, admm_max - admm], fmt="ro", capsize=6, label=r"$f(x_{\texttt{ADMM-Q}})$", linewidth=1.25, elinewidth=1.25)
ax.errorbar(x_axis, np.array(admmr), xerr=None, yerr=[admmr - admmr_min, admmr_max - admmr], fmt="bo", capsize=6, label=r"$f(x_{\texttt{ADMM-R}})$", linewidth=1.25, elinewidth=1.25)
ax.legend(loc="upper right")
ax.set_xticks(x_axis)


ax = fig.add_subplot(2, 2, 3)
x_axis = D
ax.errorbar(x_axis, np.array(admm), xerr=None, yerr=[admm - admm_min, admm_max - admm], fmt="ro", capsize=6, label=r"$f(x_{\texttt{ADMM-Q}})$", linewidth=1.25, elinewidth=1.25)
ax.errorbar(x_axis, np.array(pgd), xerr=None, yerr=[pgd - pgd_min, pgd_max - pgd], fmt="bo", capsize=6, label=r"$f(x_{\mbox{PGD}})$", linewidth=1.25, elinewidth=1.25)
ax.legend(loc="upper right")
ax.set_xticks(x_axis)

ax = fig.add_subplot(2, 2, 4)
x_axis = D
Q = []
b = []
cont = []
contp = []
for i in best_idx:
	script_dir = os.path.dirname(__file__)
	fname = "QB" + "soft_%d_%d_%d.tmp" % (d, v, i) + ".pkl"
	[Q_, b_, x0_] = pickle.load(open(fname, "rb"))
	Q = Q_[0]
	b = b_[0]
	x = -np.linalg.inv(Q) @ b
	cont.append(obj_val(Q, b, x)[0][0])
	x = my_rint(x)
	contp.append(obj_val(Q, b, x)[0][0])

ax.errorbar(x_axis, np.array(admm), xerr=None, yerr=[admm - admm_min, admm_max - admm], fmt="ro", capsize=6, label=r"$f(x_{\texttt{ADMM-Q}})$", linewidth=1.25, elinewidth=1.25)
ax.errorbar(x_axis, np.array(contp), xerr=None, yerr=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], fmt="bo", capsize=6, label=r"$f(x_{\mbox{GD}+\mbox{Proj}})$", linewidth=1.25, elinewidth=1.25)

ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("plot.png")
plt.show()
exit(0)

