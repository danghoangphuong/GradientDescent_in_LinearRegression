import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.animation as animation

def figure():
	fig = plt.figure("Gradient descent in Linear Regression")
	ax = plt.axes(xlim=(-1, 10), ylim=(-1, 8))
	return fig, ax

def GD_line(x0_lr):
	x0_gd = np.array([[0.], [1.5]])
	y_gd = x0_gd[0][0] + x0_gd[1][0]*x0_lr
	plt.plot(x0_lr, y_gd, color="black", label="GD line")
	return x0_gd

def gradient_descent(x0_gd, iteration, learning_rate, matrix_1, matrix_2):
	x_list = [x0_gd] # weight, bias list each iter
	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(x_list[-1], matrix_1, matrix_2) # x0 - alpha*f'(x0) -> [b, w]
		if np.linalg.norm(grad(x_new, matrix_1, matrix_2))/len(x_new) < 0.1: # stop gd
			print("GD stop after {} iteration".format(i))
			break
		x_list.append(x_new)
	return x_list

def grad(x, matrix_1, matrix_2): # derevative calc
	m = matrix_1.shape[0] # number of data points
	return 1/m * matrix_1.T.dot(matrix_1.dot(x) - matrix_2)

#check the derivative
def check_grad(x,maxtrix_1,matrix_2):
	eps = 1e-4
	g = np.zeros_like(x)
	for i in range(len(x)):
		x1 = x.copy()
		x2 = x.copy()
		x1[i] += eps
		x2[i] -= eps
		g[i] = (cost(x1,maxtrix_1,matrix_2) - cost(x2,maxtrix_1,matrix_2)) / (2*eps)
	g_grad = grad(x,maxtrix_1,matrix_2)
	if np.linalg.norm(g - g_grad) > 1e-5:
		print("WARNING: CHECK GRADIENT FUNCTION")

# draw procedure of GD
def	gd_procedure(x0_gd,x0_lr,maxtrix_1,matrix_2,):
	iteration = 200
	learning_rate = 1e-3
	x_list = gradient_descent(x0_gd, iteration, learning_rate, maxtrix_1, matrix_2)
	for i in range(len(x_list)):
		y_GD = x_list[i][0][0] + x_list[i][1][0]*x0_lr
		plt.plot(x0_lr, y_GD,color='black',alpha=0.5)
	return x_list

def cost(x,maxtrix_1,matrix_2):
	m = maxtrix_1.shape[0]
	return (0.5/m) * np.linalg.norm(maxtrix_1.dot(x) - matrix_2,2)**2

def cost_fig(x_list, A, b):
	cost_list = []
	iter_list = []
	for i in range(len(x_list)):
		cost_list.append(cost(x_list[i], A, b))
		iter_list.append(i)
	plt.plot(iter_list, cost_list)
	plt.xlabel("Iteration")
	plt.ylabel("Cost")
	plt.show()

def main():
	def update_line_pos(i):
		y_ani = x_list[i][0][0] + x_list[i][1][0]*x0_lr
		line.set_data(x0_lr, y_ani)
		return line,

	df = pd.read_csv("income.csv")
	
	fig, ax = figure()
	A = df.values[:, 1] #feature
	b = df.values[:, 2] #target

	plt.plot(A, b, 'ro', label="Data point")

	A = np.array([A]).T
	b = np.array([b]).T

	one = np.ones(A.shape, dtype=np.int8)
	A = np.concatenate((one, A), axis=1)

	#LR formula
	x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b) # -> [b a]

	# LR line
	x0_lr = np.linspace(1, 8, 2)
	y0_lr = x[0][0] + x[1][0]*x0_lr # b + ax
	plt.plot(x0_lr, y0_lr, linewidth=2, label="LR line")
	
	x0_gd = GD_line(x0_lr)
	check_grad(x0_gd, A, b)
	x_list = gd_procedure(x0_gd, x0_lr, A, b)

	line , = ax.plot([], [], color="yellow")
	iters = np.arange(1, len(x_list), 1)
	line_animation = animation.FuncAnimation(fig, update_line_pos, iters, interval=30, blit=True)

	plt.legend(loc="upper left")
	plt.show()

	# cost figure
	cost_fig(x_list, A, b)
	
main()