import numpy as np
import GPy
import matplotlib.pyplot as plt

from time import time
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
import random as r
import os
import sys

np.random.seed()

class BayeOpt():
    def __init__(self, target_func, x_range, init_points=3, render=True, save=False):
        self.target_func = target_func
        self.x_min = np.array(x_range[0]).reshape(-1, )  # assume x y are all continuous first
        self.x_max = np.array(x_range[1]).reshape(-1, )  # x_min and max 's shape == (self.dim,1)
        # print('xmin',self.x_min)
        # print('xmax', self.x_max)
        self.x_mid = 0.5 * (self.x_min + self.x_max)
        self.init_points = init_points
        # self.x = np.vstack((self.x_min, self.x_mid, self.x_max))[:, np.newaxis]
        self.dim = self.x_min.size
        # initialization
        self.x = np.linspace(self.x_min, self.x_max, init_points)
        # self.x = np.zeros((init_points, self.dim))
        # for i in range(init_points):
        #     r.seed()
        #     self.x[i][0] = r.uniform(0, 1)
        #     r.seed()
        #     self.x[i][1] = r.uniform(0, 1)
        print('x_in',self.x)
        self.y = target_func(np.array(self.x))
        if self.y.ndim == 1: self.y = self.y[:,None]
        self.sort_index = np.argsort(self.y,axis=0)
        # self.y_sort = np.sort(self.y,axis=0)
        print(self.y)

        self.x_range = np.linalg.norm(self.x_min - self.x_max)
        self.prior = {'sigma_nu': 0.1, 'theta0': 1, 'theta': 1, 'gammaexpectation': 2., "gammavariance": 2.}  # add some initialization strategy, theta for D length scale theta1:D
        self.render = render
        self.save = save
        self.model = None
        self.hmc_samples = None


    def constructmodel_mcmc(self):
        kernel = GPy.kern.Matern52(input_dim=self.dim, variance=1., lengthscale=1.)
        # print('y',self.y)
        # print('ydim',self.y.ndim)
        # print('x',self.x)
        # print('xdim',self.x.ndim)
        self.model = GPy.models.GPRegression(self.x, self.y, kernel)
        self.model.kern.set_prior(GPy.priors.Gamma.from_EV(self.prior['gammaexpectation'], self.prior['gammavariance']))
        self.model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(self.prior['gammaexpectation'], self.prior['gammavariance']))
        self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        self.model.optimize(max_iters=200)
        self.model.param_array[:] = self.model.param_array * (1. + np.random.randn(self.model.param_array.size) * 0.01)

        # model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        sampler = GPy.inference.mcmc.HMC(self.model, stepsize=1e-1)
        # GPy.inference.mcmc.HMC(self.model, stepsize=self.step_size)
        n_burnin = 100
        n_samples = 60
        subsample_interval = 10
        leapfrog_steps = 20
        # n_burnin = 5
        # n_samples = 5
        # subsample_interval = 1
        # leapfrog_steps = 3
        ss = sampler.sample(num_samples=n_burnin + n_samples * subsample_interval, hmc_iters=leapfrog_steps)
        self.hmc_samples = ss[n_burnin::subsample_interval]

    def predict_mcmc(self, X):
        """
        Predictions with the model for all the MCMC samples. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        input one point X to be evaluated
        return for each theta hyperparameter, the conjectural mean and std at X
        """

        if X.ndim == 1: X = X[None, :]
        ps = self.model.param_array.copy()
        means = []
        stds = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            m, v = self.model.predict(X)
            means.append(m)
            stds.append(np.sqrt(np.clip(v, 1e-10, np.inf)))
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()
        # print('means',means)
        # print('stds',stds)
        return means, stds

    def compute_integrated_ei(self, xnew):
        """
        Integrated Expected Improvement
        """
        # print('xnew.ndim',xnew.ndim)

        means, stds = self.predict_mcmc(np.array(xnew))
        f_best = [self.y.min()]
        # print('y',self.y)
        # print('f_best',f_best)

        f_acqu = 0
        for m,s,f_best in zip(means, stds, f_best):
            gamma0 = (f_best - m) / s
            f_acqu += s * (norm.cdf(gamma0) * gamma0 + norm.pdf(gamma0))
        return -(f_acqu/(len(means)))[0][0]

    def constructmodel_opt(self):
        kernel = GPy.kern.Matern52(input_dim=self.dim, variance=1., lengthscale=1.)
        # print('y',self.y)
        # print('ydim',self.y.ndim)
        # print('x',self.x)
        # print('xdim',self.x.ndim)
        self.model = GPy.models.GPRegression(self.x, self.y, kernel)

        self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)

        self.model.optimize(optimizer = 'lbfgs',max_iters=200)


    def predict_opt(self, X):
        """
        Predictions with the model for all the MCMC samples. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        input one point X to be evaluated
        return for each theta hyperparameter, the conjectural mean and std at X
        """
        if X.ndim == 1: X = X[None, :]
        m, v = self.model.predict(X)
        return m, np.sqrt(v)

    def compute_ei(self, xnew):
        means, stds = self.predict_opt(np.array(xnew))
        f_best = [self.y.min()]

        gamma0 = (f_best - means) / stds
        f_acqu = stds * (norm.cdf(gamma0) * gamma0 + norm.pdf(gamma0))
        return -f_acqu [0][0]

    def sampling(self, mode="MCMC", acquisition="EI", num=0, max_iter=10, x0=[0,0]):
        print("\n-----------------------------------------------\n")

        bounds = [(low, high) for low, high in zip(self.x_min, self.x_max)]
        if mode=="MCMC" and acquisition=="EI":
            self.constructmodel_mcmc()
            # result = minimize(self.compute_integrated_ei, x0=self.x_mid, bounds=bounds, method='Powell')
            # x_next = result.x
            # if flag==0:
            #     x0 = self.x[np.argmin(self.y)]
            # else:
            #     r.seed()
            #     x0 = np.zeros((1, 2))
            #     x0[0][0] = r.uniform(0, 1)
            #     x0[0][1] = r.uniform(0, 1)
            #
            result = fmin_l_bfgs_b(self.compute_ei, x0=x0, bounds=bounds, approx_grad=True, maxiter=15000)
            # print(result[0])
            x_next = result[0]
        elif mode=="OPT" and acquisition=="EI":
            self.constructmodel_opt()

            r.seed()
            # x0 = np.zeros((1,2))
            # x0[0][0] = r.uniform(0,1)
            # x0[0][1] = r.uniform(0, 1)
            # result = minimize(self.compute_ei, x0=self.x_mid, bounds=bounds, method='Powell')
            # x_next = result.x
            # x0=self.x_mid
            result = fmin_l_bfgs_b(self.compute_ei, x0=x0, bounds=bounds, approx_grad=True, maxiter=15000)
            # print(result[0])
            x_next = result[0]
        else:
            print('No such mode or acquisition function!')

        print("  x_new: ", x_next)
        # print('time: ', t1 - t0)
        if self.render == True:
            if self.dim==1:
                self.render_function(self.target_func, x_next, num)
            elif self.dim==2:
                if num==max_iter-1:
                    self.render_function2D(self.target_func, x_next, num)
            else:
                print('This program can only render for one or two dimensional x!')
        return x_next

    def iteration(self, x_existing=None, y_existing=None, mode="MCMC", acquisition="EI", max_iter=10):
        if x_existing != None and y_existing != None:
            self.x = x_existing
            self.y = y_existing
        ymin=np.zeros(max_iter)
        i=0
        count = 0
        t2 = time()
        x0=self.x_mid
        while 1:

            print('  # iteration: ', i)
            print('  current ymin ', self.y.min())

            # if count<min(6, len(self.y)/2):
            # if count < 1:
            #     if np.linalg.norm(self.x[self.sort_index[count]]-self.x[self.sort_index[count+1]])>np.linalg.norm(self.x_min-self.x_max)/50:
            #         x0 = (self.x[self.sort_index[count]]+self.x[self.sort_index[count+1]])/2
            #
            # else:
            if count>=1:
                print('  Random initial points.')
                r.seed()
                x0 = np.zeros((1, 2))
                x0[0][0] = r.uniform(0, 1)
                x0[0][1] = r.uniform(0, 1)
            print('  x0',x0)
            x_new = self.sampling(acquisition=acquisition, mode=mode, num=i, max_iter=max_iter,x0=x0)
            # print('  x new: ',x_new)
            if x_new in self.x:
                print("  x new is old! ")
                count+=1
            else:
                count = 0
                ymin[i] = self.y.min()
                # self.render_function2D(self.target_func, x_new)
                if x_new.ndim == 1: x_new = np.array([x_new])
                y_new = self.target_func(x_new)
                print('  y new:',y_new)
                self.x = np.vstack((self.x, x_new))
                self.y = np.vstack((self.y, [y_new]))
                self.sort_index = np.argsort(self.y,axis=0)
                # self.y_sort = np.sort(self.y,axis=0)
                print("\ny:", self.y)
                t3 = time()
                print('  time: ', t3 - t2)
                t2 = t3
                i+=1
                if i==max_iter:
                    break

        return self.x[np.argmin(self.y)],ymin

    def render_function(self, func, point, num):
        x = np.linspace(self.x_min, self.x_max, 30)
        y = func(x)
        plt.figure(figsize=(3, 3))
        plt.xlim(self.x_min, self.x_max)
        plt.plot(x, y, c="k", label="black box function")
        plt.scatter(self.x, self.y, marker='o', color="b", label="existing")
        plt.plot([point], [func(point)], marker='o', color="red", label="next")
        plt.legend(loc="best")
        if self.save == True:
            plt.savefig("./graph/func2_"+str(num)+".png")
        plt.show()

    def render_function2D(self, func, point, num=0):
        pointsnum = 200
        x = np.linspace(self.x_min, self.x_max, pointsnum )
        x1, x2 = np.meshgrid(x[:, 0], x[:, 1])
        y = np.zeros((pointsnum , pointsnum) )
        h = np.zeros((pointsnum , 2))
        for i in range(pointsnum):
            h[:, 0] = x1[:, i]
            h[:, 1] = x2[:, i]
            # print('h', h)
            y[:, i] = func(h)

        fig, ax = plt.subplots()
        pcm = ax.pcolor(x1, x2, y)
        fig.colorbar(pcm, ax=ax, extend='max')

        plt.scatter(self.x[:,0], self.x[:,1], marker='o', color="m", label="existing")
        print('point',point)
        plt.plot([point[0]], [point[1]], marker='o', color="r", label="next")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(bbox_to_anchor=(1.02, 1.12), loc='upper left', borderaxespad=0)
        if self.save == True:
            plt.savefig("./graph/func2_opteitest" + str(num) + ".png")
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")


class FileOperation():
    def __init__(self, filename):
        self.filename = filename

    def checkfile(self):
        if os.path.isfile(self.filename)==True:
            print('File already exists.')
            while 1:
                a = input('Which one do you want?\n1. Wipe the file and continue\n2. Append.\n3. Exit. [1/2/3] ')
                if a == '1':
                    with open(self.filename, 'w') as file:
                        pass
                    print('Finish wiping the file.')
                    return
                elif a == '2':
                    return
                elif a == '3':
                    sys.exit("Exit the program now. Please change the file name:)")
                else:
                    print('Wrong input. Please type [1/2/3] again.')
        else:
            with open(self.filename, 'w') as file:
                pass
            print('Finish creating the file.')

    def readdata(self):
        with open(self.filename, 'r') as fo:
            s = fo.read()
        s = s.replace("\n", "")
        s = s[1:len(s) - 1]
        s = s.split('][')
        for i in range(len(s)):
            s[i] = s[i].lstrip()
            s[i] = s[i].rstrip()
            s[i] = s[i].split()
            s_floats = [float(x) for x in s[i]]
            s[i] = s_floats
        # print(s)
        return s

    def plotdata(self, ymin):
        m = np.mean(ymin, axis=0)
        s = np.std(ymin, axis=0)
        # print(m, s)
        plt.errorbar(range(1, len(m) + 1), m, yerr=s, fmt='-o')

        plt.xlabel('number of iterations')
        plt.ylabel('$y_{min}$')
        plt.show()

    def plotconvergence(self):
        y = self.readdata()
        self.plotdata(y)


def main():
    # reminder for file name change, in case overwriting
    print('Remember to change the file name before running this program!')
    while 1:
        a = input('Do you want to start running? [y/n] ')
        if a.lower() == 'y':
            break
        elif a.lower() == 'n':
            print('Exit the program now. Please change the file name:)')
            return
        else:
            print('Wrong input. Please type [y/n] again.')

    txtflag = 0
    while 1:
        a = input('Do you want to save data to txt? [y/n] ')
        if a.lower() == 'y':
            txtflag = 1
            break
        elif a.lower() == 'n':
            break
        else:
            print('Wrong input. Please type [y/n] again.')

    # whether the txt file already exists
    if txtflag:
        txtfile = 'func2_opt0505_100.txt'
        file = FileOperation(filename=txtfile)
        file.checkfile()

    # problem setting
    # black box function definition
    # g = lambda x:(6*x-2)**2*np.sin(12*x-4)
    # g_range = [0,1]
    # g = lambda x:np.sin(x)
    # g_range = [0,10]
    # g = lambda x:x**2
    # g_range = [-5,2]
    g = lambda x: (np.square(
        (x[:, 1] * 15) - 5 - (5.1 / (4 * np.square(np.pi))) * np.square(x[:, 0] * 15) + (5 / np.pi) * x[:,
                                                                                                      0] * 15 - 6) + 10 * (
                               1 - (1. / (8 * np.pi))) * np.cos(x[:, 0] * 15) + 10)
    g_range = [[0, 0], [1, 1]]  # first row: all the xmin, second row: all the xmax

    init_point = 4
    evaluation = 100
    # evaluation = 6
    repeat = 10
    ymin = np.zeros((repeat, evaluation - init_point))


    for k in range(repeat):
        print('# of repeat: ', k)
        t0 = time()
        baye_GP_ei = BayeOpt(target_func=g, x_range=g_range, init_points=init_point, render=False, save=False)
        best_result, ymin[k, :] = baye_GP_ei.iteration(mode="OPT", acquisition="EI", max_iter=evaluation - init_point)

        if txtflag:
            with open(txtfile, 'a') as fo:
                fo.write(str(ymin[k, :]) + '\n')

        t1 = time()
        print('time: ', t1 - t0)

        plt.plot(range(1, evaluation - init_point + 1), ymin[k, :])
        plt.xlabel('number of iterations')
        plt.ylabel('$y_{min}$')
        plt.title('# of repeat: ' + str(k))
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")
    print('End of the test.\n\n')

    plot_flag = 0
    while 1:
        a = input('Do you want to save the convergence graph? [y/n] ')
        if a.lower() == 'y':
            plot_flag = 1
            while 1:
                plotfile = input('Please enter the plot file path: ')
                if os.path.isfile(plotfile) == True:
                    print('File already exists.')
                else:
                    break
            break
        elif a.lower() == 'n':
            break
        else:
            print('Wrong input. Please type [y/n] again.')

    m = np.mean(ymin, axis=0)
    s = np.std(ymin, axis=0)
    # print(m, s)
    plt.errorbar(range(1, evaluation - init_point + 1), m, yerr=s, fmt='-o')

    plt.xlabel('number of iterations')
    plt.ylabel('$y_{min}$')
    if plot_flag==1:
        plt.savefig(plotfile)
    plt.show()
    print("\n\n\n-----------------------------------------------\n\n\n")
    print("best result is ", best_result)
    print('Finished!')
    print("\n\n\n-----------------------------------------------")
    print("-----------------------------------------------\n\n\n")


    # whether plot the convergence graph
    if txtflag:
        while 1:
            a = input('Do you want to plot the convergence graph? [y/n] ')
            if a.lower() == 'y':
                file.plotconvergence()
                print('Finish plotting the convergence graph.\nBye!')
                return
            elif a.lower() == 'n':
                print('Exit the program now.\nBye!')
                return
            else:
                print('Wrong input. Please type [y/n] again.')
    else:
        print('Exit the program now.\nBye!')


if __name__ == "__main__":
    main()

