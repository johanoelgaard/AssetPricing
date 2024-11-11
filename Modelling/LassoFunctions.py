import numpy as np 
import scipy.stats as stats
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import copy

def standardize(X):

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_stan = (X - mu) / sigma

    return X_stan

def cv_pen_grid(range=[0.01,80000], candidates=50):
    # create log-spaced grid of penalty values
    penalty_grid = np.geomspace(range[0], range[1], num=candidates)

    return penalty_grid

def cv_pen(X,y,pen_grid, do_print=False):
    # estimate penalty using 5-fold cross-validation
    fit_CV = lm.LassoCV(cv=5, alphas=pen_grid).fit(X,y)
    penalty_cv = fit_CV.alpha_

    if do_print:
        print('Penalty_CV: ', round(penalty_cv, 2))

    return penalty_cv, fit_CV

def brt_pen(X_tilde,y,c=1.1,alpha=0.05,do_print=False):
    n,p = X_tilde.shape
    sigma = np.std(y, ddof=1)
    max_term = np.max((1/n) * np.sum((X_tilde**2),axis=0))**0.5 # Note: this equals 1 for standardized data, and is therefore not strictly necessary when using standardized data
    penalty_BRT = (c * sigma) / np.sqrt(n) * stats.norm.ppf(1 - alpha / (2*p))*max_term

    if do_print:
        print('Max term: ', round(max_term, 2))
        print('Penalty_BRT: ', round(penalty_BRT, 2))

    return penalty_BRT



def bcch_pen_pilot(X,y,c=1.1,alpha=0.05,do_print=False):
    n,p = X.shape
    
    yXscale = (np.max((X.T ** 2) @ ((y-np.mean(y)) ** 2) / n)) ** 0.5
    penalty_pilot = c / np.sqrt(n) * stats.norm.ppf(1-alpha/(2*p)) * yXscale

    if do_print:
        print('Penalty_pilot: ', round(penalty_pilot, 2))

    return penalty_pilot

def bcch_pen(X, y, c=1.1, alpha=0.05, do_print=False):
    n,p = X.shape
    penalty_pilot = bcch_pen_pilot(X,y,c=c,alpha=alpha)
    pred = lm.Lasso(alpha=penalty_pilot).fit(X, y).predict(X)
    eps = y - pred
    epsXscale = np.sqrt(np.max((X.T**2 @ eps**2)/n))
    penalty_BCCH = c / np.sqrt(n) * stats.norm.ppf(1-alpha/(2*p))*epsXscale

    if do_print:
        print('Penalty_BCCH: ', round(penalty_BCCH, 2))
    
    return penalty_BCCH

def alpha_pdl(resid: dict, d, do_print=False):
    # check that the right keys are present

    for keys in ['resyxz', 'resdz']:
        if keys not in resid.keys():
            raise ValueError(f"Key {keys} not present in resid dictionary")
    
    # unpack residuals
    resyxz = resid['resyxz']
    resdz = resid['resdz']

    # calculate alpha
    num = resdz.T @ resyxz
    denom = resdz.T @ d
    alpha = num/denom

    if do_print:
        print('alpha PDL: ', alpha[0][0].round(6))

    return alpha[0][0]

def se_pdl(resid: dict, do_print=False):
    # check that the right keys are present

    for keys in ['resyx', 'resdz']:
        if keys not in resid.keys():
            raise ValueError(f"Key {keys} not present in resid dictionary")
    
    # unpack residuals
    resyx = resid['resyx']
    resdz = resid['resdz']

    N = resdz.shape[0]

    # Calculate variance    
    num = resdz.T**2@resyx**2/N
    denom = (resdz.T@resdz/N)**2
    sigma2_PDL = num/denom

    # Calculate standard error
    se = np.sqrt(sigma2_PDL/N)

    if do_print:
        print('Standard error PDL: ', se[0][0].round(6))
        # print('Variance PDL: ', sigma2_PDL[0][0].round(6))

    return se[0][0], sigma2_PDL[0][0]

def confidence_pdl(alpha, se, p=0.05, do_print=False):
    # calculate confidence interval
    q = stats.norm.ppf(1-p/2)
    lower = alpha - q * se
    upper = alpha + q * se

    if do_print:
        print('Confidence interval: ', (lower.round(6), upper.round(6)))

    return lower, upper


def plot_lasso_path(penalty_grid, coefs, legends, text_height, vlines: dict = None, do_print=False, save_path=None):
    """
    Plots the coefficients as a function of the penalty parameter for Lasso regression.

    Parameters:
    penalty_grid (array-like): The penalty parameter values.
    coefs (array-like): The estimated coefficients for each penalty value.
    legends (list): The labels for each coefficient estimate.
    vlines (dict, optional): A dictionary of vertical lines to add to the plot. The keys are the names of the lines and the values are the penalty values where the lines should be drawn.
    
    """
    # Initiate figure 
    fig, ax = plt.subplots()

    # Plot coefficients as a function of the penalty parameter
    ax.plot(penalty_grid, coefs)

    # Set log scale for the x-axis
    ax.set_xscale('log')

    # Add labels
    plt.xlabel('Penalty, $\lambda$')
    plt.ylabel(r'Estimates, $\widehat{\beta}_j(\lambda)$')
    # plt.title('Lasso Path')

    # Add legends
    if legends is not None:
        lgd=ax.legend(legends,loc=(1.04,0))
    
    # Add vertical lines
    if vlines is not None:
        for name, penalty in vlines.items():
            ax.axvline(x=penalty, linestyle='--', color='grey')
            plt.text(penalty,text_height,name,rotation=90)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight',dpi=300)
        print(f"Figure saved to {save_path}")
    
    # Display plot
    if do_print:
        plt.show()
    else:
        plt.close(fig)


def plot_MSE_path(penalty_grid, MSE, text_height, do_print=False, save_path=None):
    """
    Plots the mean squared error (MSE) as a function of the penalty parameter.

    Parameters:
    penalty_grid (array-like): The penalty parameter values.
    MSE (array-like): The corresponding MSE values.

    """
    # Initiate figure 
    fig, ax = plt.subplots()

    # Plot MSE as a function of the penalty parameter
    ax.plot(penalty_grid, MSE)

    # Set log scale for the x-axis
    ax.set_xscale('log')
    
    # Plot minimum MSE
    min_MSE_idx = np.argmin(MSE)
    min_MSE_penalty = penalty_grid[min_MSE_idx]
    ax.axvline(x=min_MSE_penalty, linestyle='--', color='grey')
    plt.text(min_MSE_penalty,text_height,"Minimum MSE",rotation=90)

    # Add labels
    plt.xlabel('Penalty, $\lambda$')
    plt.ylabel('Mean squared error')
    # plt.title('Mean squared error')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight',dpi=300)
        print(f"Figure saved to {save_path}")
    
    # Display plot
    if do_print:
        plt.show()
    else:
        plt.close(fig)


def generate_table(
    results_dict: dict,
    filename: str,
) -> None:
    """
    Exports regression results to a LaTeX table and saves it as a .txt file.

    Args:
        results_dict (dict): List of result dictionaries from the estimate function.
        filename (str): The filename to save the LaTeX table (should end with .txt).
    """

    dict_ = copy.deepcopy(results_dict)

    cols = list(dict_.keys())
    num_models = len(cols)
    
    # Add significance stars to the results
    for col in dict_:
        result = dict_[col][0]
        std_error = dict_[col][1]
        
        # Calculate the z-score
        z_score = result / std_error
        
        # Calculate the p-value from the z-score
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Determine the significance level and add stars
        if p_value < 0.01:
            significance = '***'
        elif p_value < 0.05:
            significance = '**'
        elif p_value < 0.1:
            significance = '*'
        else:
            significance = ''
        
        # Append the significance stars to the result
        dict_[col].append(significance)
    
    # Start constructing the LaTeX table
    lines = []

    lines.append("\\begin{tabular}{" + "l" + "c" * num_models + "}")
    lines.append("\\hline\\hline\\\\[-1.8ex]")
    header_row = [""] + cols
    lines.append(" & ".join(header_row) + " \\\\")
    lines.append("\\hline")

    # For each variable in label_x
    lines.append("$\\hat{\\beta}$ & " + " & ".join([f"{dict_[col][0]:.4f}"+f"{dict_[col][2]}" for col in cols]) + " \\\\")
    lines.append(" & " + " & ".join([f"({dict_[col][1]:.4f})" for col in cols]) + " \\\\")

    lines.append("\\hline\\hline")

    # End of table
    lines.append("\\end{tabular}")

    # Save to file
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

    print(f"LaTeX table saved to {filename}")