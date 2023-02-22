import numpy as np 
from numba import jit
import matplotlib.pyplot as plt
import sys
import os
from data_writer import *
import pickle

def my_nan_to_num(tmp):

	for i in range(len(tmp)):
		if np.isnan(tmp[i]):
			tmp[i] = 1e307
		if np.isinf(tmp[i]):
			tmp[i] = 1e307
		if np.isneginf(tmp[i]):
			tmp[i] = -1e307

	return tmp

@jit(nopython=True, cache=True)
def my_rint(x, delta_=8.0):

	return np.rint(x / delta_) * delta_

@jit(nopython=True, cache=True)
def obj_val(Q, b, x):

	return 0.5 * (x.T @ Q) @ x + b.T @ x

@jit(nopython=True, cache=True)
def my_mod(a, b):
	c = np.floor(a / b)
	return a - b * c

@jit(nopython=True, cache=True)
def my_bernoulli(shape, p):
	# return (np.random.random(shape) < p).astype(np.float)
	return (np.random.random(shape) < p) * 1.0

@jit(nopython=True, cache=True)
def PGD(Q_, b_, x0_, rho_, D):

	np.random.seed(100)
	N, _, _ = Q_.shape
	N_iter = 100000
	N_last = 50
	results = np.zeros(N)

	for i in range(N):
		
		Q = Q_[0]
		b = b_[0]
		x = x0_[i]

		val, vec = np.linalg.eig(Q)
		L = np.max(val)
		rho = rho_ * L

		x_all = np.zeros(N_last)

		for j in range(N_iter):

			x = my_rint(x - (1 / rho) * (Q @ x + b));

			if j >= N_iter - N_last:
				x_all[j - (N_iter - N_last)] = obj_val(Q, b, x)[0][0]

		results[i] = np.min(x_all)

	return results

@jit(nopython=True, cache=True)
def ADMMr(Q_, b_, x0_, noise_power, rho_, D):

	np.random.seed(300)
	N, _, _ = Q_.shape
	N_iter = 30000
	N_last = 50
	results = np.zeros(N)
	S = 0

	for i in range(N):

		Q = Q_[0]
		b = b_[0]
		x = x0_[i]

		val, vec = np.linalg.eig(Q)
		L = np.max(val)
		rho = rho_ * L

		lambda_ = np.zeros((D, 1))

		x_all = np.zeros(N_last)
		x_y = my_rint(x + lambda_ / rho)

		Q_inv = np.linalg.inv(Q + rho * np.eye(D))

		for j in range(N_iter):

			mask = np.random.random(x.shape) < noise_power 

			x_y = my_rint(x + lambda_ / rho) * mask + x_y * (1 - mask)
			
			x = Q_inv @ (rho * x_y - b - lambda_)
			lambda_ = lambda_ + rho * (x - x_y)

			if j >= N_iter - N_last:
				x_all[j - (N_iter - N_last)] = obj_val(Q, b, my_rint(x_y))[0][0]
		

		results[i] = np.min(x_all)

	return results

@jit(nopython=True, cache=True)
def ADMMs(Q_, b_, x0_, noise_power, rho_, D):

	np.random.seed(300)
	N, _, _ = Q_.shape
	N_iter = 30000
	N_last = 50
	results = np.zeros(N)
	S = 0

	for i in range(N):

		Q = Q_[0]
		b = b_[0]
		x = x0_[i]

		val, vec = np.linalg.eig(Q)
		L = np.max(val)
		rho = rho_ * L

		lambda_ = np.zeros((D, 1))

		x_all = np.zeros(N_last)

		Q_inv = np.linalg.inv(Q + rho * np.eye(D))

		for j in range(N_iter):

			z = x + lambda_ / rho
			z_ = my_rint(z)
			beta_ = noise_power / rho

			z_dist_norm = np.linalg.norm(z - z_)

			if beta_ < z_dist_norm:
				x_y = z + beta_ * (z_ - z) / z_dist_norm
			else:
				x_y = z_
			
			x = Q_inv @ (rho * x_y - b - lambda_)
			lambda_ = lambda_ + rho * (x - x_y)

			if j >= N_iter - N_last:
				x_all[j - (N_iter - N_last)] = obj_val(Q, b, my_rint(z_))[0][0]
		

		results[i] = np.min(x_all)

	return results

@jit(nopython=True, cache=True)
def ADMM(Q_, b_, x0_, rho_, D):

	np.random.seed(300)
	N, _, _ = Q_.shape
	N_iter = 30000
	N_last = 50
	results = np.zeros(N)

	for i in range(N):

		Q = Q_[0]
		b = b_[0]
		x = x0_[i]

		val, vec = np.linalg.eig(Q)
		L = np.max(val)
		rho = rho_ * L

		lambda_ = np.zeros((D, 1))

		x_all = np.zeros(N_last)
		
		Q_inv = np.linalg.inv(Q + rho * np.eye(D))

		for j in range(N_iter):

			x_y = my_rint(x + lambda_ / rho)
			x = Q_inv @ (rho * x_y - b - lambda_)
			lambda_ = lambda_ + rho * (x - x_y)


			if j >= N_iter - N_last:
				x_all[j - (N_iter - N_last)] = obj_val(Q, b, x_y)[0][0]

		results[i] = np.min(x_all)


	return results

# @jit
def run_helper(soft_p, random_p, rhos, N_trails, D, fname, v):

	N_s, = soft_p.shape
	N_r, = random_p.shape
	N_j, = rhos.shape

	opt_ADMM = np.zeros((7 + N_trails, N_s, N_j))
	opt_ADMMr = np.zeros((7 + N_trails, N_s, N_j))
	opt_ADMMs = np.zeros((7 + N_trails, N_s, N_j))
	opt_PGD = np.zeros((7 + N_trails, N_s, N_j))

	Q_ = np.zeros((N_trails, D, D))
	b_ = np.zeros((N_trails, D, 1))
	x0_ = np.zeros((N_trails, D, 1))

	opt = 0.0
	opt_D = 0.0
	opt_Z = 0.0
	for i in range(N_trails):
		tmp1 = np.random.randn(D, D)
		tmp2 = v * np.random.randn(D, 1) 
		Q_[i] = tmp1.T @ tmp1 + tmp2 @ tmp2.T
		b_[i] = np.random.randn(D, 1) * 20

		x0_[i] = np.random.randn(D, 1) * 10


		opt_cont = -np.linalg.inv(Q_[i]) @ b_[i]

		print(np.linalg.cond(Q_[i]))
		if i == 0:
			opt = opt + obj_val(Q_[i], b_[i], opt_cont)[0][0]
			opt_D = opt_D + obj_val(Q_[i], b_[i], my_rint(opt_cont))[0][0]
			opt_Z = opt_Z + obj_val(Q_[i], b_[i], np.zeros((D, 1)))[0][0]


	print("Best possible =", opt)
	print("Best possible D =", opt_D)
	print("Best possible Z =", opt_Z)

	pickle.dump([Q_, b_, x0_], open("QB" + fname + ".pkl", "wb"))

	for i in range(N_s):
		for j in range(N_j):
			print(i, j)

			tmp = ADMMs(Q_, b_, x0_, soft_p[i], rhos[j], D)
			opt_ADMMs[0][i][j], opt_ADMMs[1][i][j] = np.mean(tmp), np.std(tmp)
			opt_ADMMs[2][i][j] = np.median(tmp)
			opt_ADMMs[3][i][j] = np.min(tmp)
			opt_ADMMs[4][i][j] = np.max(tmp)
			opt_ADMMs[5][i][j] = np.quantile(tmp, 0.25)
			opt_ADMMs[6][i][j] = np.quantile(tmp, 0.75)

			opt_ADMMs[7: 7 + N_trails, i, j] = tmp 

	for i in range(N_r):
		for j in range(N_j):
			print(i, j)

			tmp = my_nan_to_num(ADMMr(Q_, b_, x0_, random_p[i], rhos[j], D))
			opt_ADMMr[0][i][j], opt_ADMMr[1][i][j] = np.mean(tmp), np.std(tmp)
			opt_ADMMr[2][i][j] = np.median(tmp)
			opt_ADMMr[3][i][j] = np.min(tmp)
			opt_ADMMr[4][i][j] = np.max(tmp)
			opt_ADMMr[5][i][j] = np.quantile(tmp, 0.25)
			opt_ADMMr[6][i][j] = np.quantile(tmp, 0.75)

			opt_ADMMr[7: 7 + N_trails, i, j] = tmp 

	for i in range(1):
		for j in range(N_j):
			print(i, j)
			tmp = ADMM(Q_, b_, x0_, rhos[j], D)
			opt_ADMM[0][i][j], opt_ADMM[1][i][j] = np.mean(tmp), np.std(tmp)
			opt_ADMM[2][i][j] = np.median(tmp)
			opt_ADMM[3][i][j] = np.min(tmp)
			opt_ADMM[4][i][j] = np.max(tmp)
			opt_ADMM[5][i][j] = np.quantile(tmp, 0.25)
			opt_ADMM[6][i][j] = np.quantile(tmp, 0.75)
			opt_ADMM[7: 7 + N_trails, i, j] = tmp 

			tmp = my_nan_to_num(PGD(Q_, b_, x0_, rhos[j], D))
			opt_PGD[0][i][j], opt_PGD[1][i][j] = np.mean(tmp), np.std(tmp)
			opt_PGD[2][i][j] = np.median(tmp)
			opt_PGD[3][i][j] = np.min(tmp)
			opt_PGD[4][i][j] = np.max(tmp)
			opt_PGD[5][i][j] = np.quantile(tmp, 0.25)
			opt_PGD[6][i][j] = np.quantile(tmp, 0.75)
			opt_PGD[7: 7 + N_trails, i, j] = tmp 

	idx = np.argmin(opt_ADMM[2][0: 1].reshape(-1))
	ADMM_best_avg = opt_ADMM[0][0: 1].reshape(-1)[idx]
	ADMM_best_min = opt_ADMM[3][0: 1].reshape(-1)[idx]
	ADMM_best_max = opt_ADMM[4][0: 1].reshape(-1)[idx]
	ADMM_best_1q = opt_ADMM[5][0: 1].reshape(-1)[idx]
	ADMM_best_2q = opt_ADMM[2][0: 1].reshape(-1)[idx]
	ADMM_best_3q = opt_ADMM[6][0: 1].reshape(-1)[idx]

	idx = np.argmin(opt_PGD[2][0: 1].reshape(-1))
	PGD_best_avg = opt_PGD[0][0: 1].reshape(-1)[idx]
	PGD_best_min = opt_PGD[3][0: 1].reshape(-1)[idx]
	PGD_best_max = opt_PGD[4][0: 1].reshape(-1)[idx]
	PGD_best_1q = opt_PGD[5][0: 1].reshape(-1)[idx]
	PGD_best_2q = opt_PGD[2][0: 1].reshape(-1)[idx]
	PGD_best_3q = opt_PGD[6][0: 1].reshape(-1)[idx]

	idx = np.argmin(opt_ADMMs[2].reshape(-1))
	ADMMs_best_avg = opt_ADMMs[0].reshape(-1)[idx]
	ADMMs_best_min = opt_ADMMs[3].reshape(-1)[idx]
	ADMMs_best_max = opt_ADMMs[4].reshape(-1)[idx]
	ADMMs_best_1q = opt_ADMMs[5].reshape(-1)[idx]
	ADMMs_best_2q = opt_ADMMs[2].reshape(-1)[idx]
	ADMMs_best_3q = opt_ADMMs[6].reshape(-1)[idx]

	idx = np.argmin(opt_ADMMr[2][0: N_r].reshape(-1))
	ADMMr_best_avg = opt_ADMMr[0][0: N_r].reshape(-1)[idx]
	ADMMr_best_min = opt_ADMMr[3][0: N_r].reshape(-1)[idx]
	ADMMr_best_max = opt_ADMMr[4][0: N_r].reshape(-1)[idx]
	ADMMr_best_1q = opt_ADMMr[5][0: N_r].reshape(-1)[idx]
	ADMMr_best_2q = opt_ADMMr[2][0: N_r].reshape(-1)[idx]
	ADMMr_best_3q = opt_ADMMr[6][0: N_r].reshape(-1)[idx]


	script_dir = os.path.dirname(__file__)
	rel_path = "tmp/" + fname
	abs_file_path = os.path.join(script_dir, rel_path)
	with open(abs_file_path, "w") as f:
		f.write(str(ADMM_best_1q) + "\n")
		f.write(str(ADMM_best_2q) + "\n")
		f.write(str(ADMM_best_3q) + "\n")
		f.write(str(ADMMs_best_1q) + "\n")
		f.write(str(ADMMs_best_2q) + "\n")
		f.write(str(ADMMs_best_3q) + "\n")
		f.write(str(ADMMr_best_1q) + "\n")
		f.write(str(ADMMr_best_2q) + "\n")
		f.write(str(ADMMr_best_3q) + "\n")
		f.write(str(PGD_best_1q) + "\n")
		f.write(str(PGD_best_2q) + "\n")
		f.write(str(PGD_best_3q) + "\n")

	print("D, V =", D, v)


	# return opt_ADMM, opt_ADMM2, opt_PGD

def run(D, v, fname):

	soft_p = np.power(10.0, np.linspace(-5, 5, 21))
	random_p = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
	rhos = np.power(10.0, np.linspace(-6, 2, 9))
	# N_row, = noises.shape
	# N_col, = rhos.shape
	N_trails = 50
	run_helper(soft_p, random_p, rhos, N_trails, D, fname, v)

	# row = 0
	# writer = DataWriter("results_soft_%d_new.xlsx" % D, "beta", "rho")
	# writer.set_col_fill("n/a")
	# writer.write_block(opt_ADMM[0], row, 0, (noises[0: 1], rhos), "ADMM mean")
	# writer.write_block(opt_ADMM[1], row, N_col + 4, (noises[0: 1], rhos), "ADMM std")
	# writer.set_col_fill("")
	# row += 6
	# writer.write_block(opt_ADMM2[0], row, 0, (noises, rhos), "ADMM soft mean")
	# writer.write_block(opt_ADMM2[1], row, N_col + 4, (noises, rhos), "ADMM soft std")
	# row += N_row + 5
	# writer.set_col_fill("n/a")
	# writer.write_block(opt_ADMM[3], row, 0, (noises[0: 1], rhos), "ADMM min")
	# writer.set_col_fill("")
	# row += 6
	# writer.write_block(opt_ADMM2[3], row, 0, (noises, rhos), "ADMM soft min")
	# row += N_row + 5
	# writer.set_col_fill("n/a")
	# writer.write_block(opt_ADMM[4], row, 0, (noises[0: 1], rhos), "ADMM max")
	# writer.set_col_fill("")
	# row += 6
	# writer.write_block(opt_ADMM2[4], row, 0, (noises, rhos), "ADMM soft max")
	# row += N_row + 5
	# writer.set_col_fill("n/a")
	# writer.write_block(opt_ADMM[2], row, 0, (noises[0: 1], rhos), "ADMM median")
	# writer.set_col_fill("")
	# row += 6
	# writer.write_block(opt_ADMM2[2], row, 0, (noises, rhos), "ADMM soft median")
	# row += N_row + 5
	# # writer.set_col_fill("n/a")
	# # writer.write_block(opt_PGD[0], row, 0, (noises[0: 1], rhos), "PGD mean")
	# # writer.write_block(opt_PGD[1], row, N_col + 4, (noises[0: 1], rhos), "PGD std")
	# # writer.set_col_fill("")


	# for i in range(N_trails):
	# 	row = 0
	# 	writer.set_col_fill("n/a")
	# 	writer.write_log(opt_ADMM[i + 7], row, 0, (noises[0: 1], rhos), "ADMM", i + 1)
	# 	writer.set_col_fill("")
	# 	row += 6
	# 	writer.write_log(opt_ADMM2[i + 7], row, 0, (noises, rhos), "ADMM soft", i + 1)

	

def main(D, v, fname):
	run(D, v, fname)


if __name__ == "__main__":
	args = []
	for i, arg in enumerate(sys.argv):
		args.append(arg)
	print(args)
	main(int(args[1]), int(args[2]), args[3])