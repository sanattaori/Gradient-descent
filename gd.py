import numpy as np
import pandas as pd

# pandas dataframe
df = pd.read_csv('A2_Q4_data.csv', delimiter=',')

# X and Y values as numpy array
X = df.iloc[:, :1].values
Y = df.iloc[:, 1:2].values


def f(w,b,x):
	return 1.0/(1.0+np.exp(-(w*x+b)))

def error(w,b):
	err = 0.0
	for x,y in zip(X,Y):
		fx = f(w,b,x)
		err += 0.5 * (fx - y) ** 2
	return err

def grad_b(w,b,x,y):
	fx = f(w,b,x)
	return (fx - y) * fx * (1 - fx)

def grad_w(w,b,x,y):
	fx = f(w,b,x)
	return (fx - y) * fx * (1 - fx) * x

def do_gradient():
	w,b,eta,max_epoches = 1,1,0.01,100

	for i in range(max_epoches):
		dw, db = 0, 0
		for x,y in zip(X, Y):
			dw += grad_w(w, b, x, y)
			db += grad_b(w, b, x, y)
		w = w - eta * dw
		b = b - eta * dw
	return w,b

def main():
	inib = 1
	iniw = 1

	print("before gradient descent at b = {0}, w = {1}, error = {2}".format(inib, iniw, error(iniw, inib)))

	w, b = do_gradient()

	print("After gradient descent at b = {0}, w = {1}, error = {2}".format(b, w, error(w, b)))

	# output 
	# before gradient descent at b = 1, w = 1, error = [0.06240145]
	# After gradient descent at b = [1.17498164], w = [1.17498164], error = [0.04973018]


if __name__ == '__main__':
	main()