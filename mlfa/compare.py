import numpy as np
from scipy import linalg
from utils import *
from emmlfa import EMMlFactorAnalysis
from ml_fa import MlFactorAnalysis
from simulate import FactorModel

def compare_ll(settings,noise_type, n_factors):
    '''
        for each setting and number pf factors, plot the LL against iterations
    '''
    lambdas, std = settings()

    fm = FactorModel(1000, lambdas, std, seed=10)
    fm.simulate()
    cyy = fm.cov()
    lambdas, taus = fm.get_pca_factors(n_factors)
    init_params = {"lambdas" : lambdas.T, "taus" : taus}
    # init_params = {}
    # init_params["lambdas"] = np.array([[0.7, 0.7, 0.3, 0.3, 0.6, 0.7, 0.6, 0.5, 0.4],
    #                                    [-0.5,-0.1,0.8, 0.7, 0.0, 0.0, 0.0, 0.6, 0.6],
    #                                    [0.1, 0.2, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # init_params["taus"] = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    emmlfa = EMMlFactorAnalysis(n_factors,1000, init_params, plot=False)
    emmlfa.fit(cyy)
    print("EM lambdas shape: %d * %d" % (emmlfa.lambdas.shape[0], emmlfa.lambdas.shape[1]))

    mlfa = MlFactorAnalysis(n_factors,1000, init_params, plot=False)
    mlfa.fit(cyy)
    print("ML lambdas shape: %d * %d" % (emmlfa.lambdas.shape[0], emmlfa.lambdas.shape[1]))

    plot_comparison(mlfa.ll, emmlfa.ll, file_name="exp_%s_%d" % (noise_type, n_factors))

def draw_lls(settings):
    for setting in settings:
        for n_factors in xrange(1,5):
            compare_ll(setting[1], setting[0], n_factors)

def batch_simulations(settings):
    '''
        for each setting and each number of factors, simulate 500 times to obtain
        the distribution of iterations and mean/std of estimators
    '''
    f = open(os.path.join('.', "result.txt"),'wb')
    for setting in settings:
        f.write("Noise type: %s \n" % setting[0])
        for n_factors in xrange(1,5):
            noise_type, setting_fun = setting
            ml_iters = []
            ml_lam_list = []
            ml_taus_list = []

            emmlfa_iters = []
            emmlfa_lam_list = []
            emmlfa_taus_list = []


            lambdas, std = setting_fun()
            for i in xrange(500):
                fm = FactorModel(1000, lambdas, std, seed=i)
                ml_iter, ml_lam, ml_taus, emmlfa_iter, emmlfa_lam, emmlfa_taus = compare_iter(fm, n_factors)
                ml_iters.append(ml_iter)
                ml_lam_list.append(ml_lam)
                ml_taus_list.append(ml_taus)

                emmlfa_iters.append(emmlfa_iter)
                emmlfa_lam_list.append(emmlfa_lam)
                emmlfa_taus_list.append(emmlfa_taus)

            # print("Draw blox plot for exp_%s_%d_boxplot" % (noise_type, n_factors))
            # draw_iter_boxplot(ml_iters, emmlfa_iters,file_name="exp_%s_%d_boxplot" % (setting[0], n_factors))
            f.write("Estimates for %d factors\n" % n_factors)
            ml_mean_lam, ml_var_lam = cal_stat(ml_lam_list)
            ml_mean_taus, ml_var_taus = cal_stat(ml_taus_list)
            em_mean_lam, em_var_lam = cal_stat(emmlfa_lam_list)
            em_mean_taus, em_var_taus = cal_stat(emmlfa_taus_list)
            f.write("Mean of lambda for ML \n")
            f.write(np.array2string(ml_mean_lam, separator=',') + '\n')
            f.write("Mean of lambda for EM \n")
            f.write(np.array2string(em_mean_lam, separator=',') + '\n')
            f.write("Variance of lambda for ML \n")
            f.write(np.array2string(ml_var_lam, separator=',') + '\n')
            f.write("Variance of lambda for EM \n")
            f.write(np.array2string(em_var_lam, separator=',') + '\n')
            #write taus
            f.write("\nMean of taus for ML \n")
            f.write(np.array2string(np.diag(ml_mean_taus), separator=',') + '\n')
            f.write("Mean of taus for EM \n")
            f.write(np.array2string(np.diag(em_mean_taus), separator=',') + '\n')
            f.write("Variance of taus for ML \n")
            f.write(np.array2string(np.diag(ml_var_taus), separator=',') + '\n')
            f.write("Variance of taus for EM \n")
            f.write(np.array2string(np.diag(em_var_taus), separator=',') + '\n')
            f.write("\n\n")
            print "%d factors finished" % n_factors
    f.close()







def compare_iter(fm,n_factors):
    fm.simulate()
    cyy = fm.cov()
    lambdas, taus = fm.get_pca_factors(n_factors)
    init_params = {"lambdas" : lambdas.T, "taus" : taus}

    emmlfa = EMMlFactorAnalysis(n_factors,1000, init_params, plot=False)
    emmlfa.fit(cyy)

    mlfa = MlFactorAnalysis(n_factors,1000, init_params, plot=False)
    mlfa.fit(cyy)
    return mlfa.iterations, mlfa.lambdas, mlfa.taus, emmlfa.iterations, emmlfa.lambdas, emmlfa.taus


if __name__ == "__main__":
    settings = [("normal", load_normal_settings),
                ("lownoise", load_lownoise_settings),
                ("highnoise", load_highnoise_settings)]

    batch_simulations(settings)

