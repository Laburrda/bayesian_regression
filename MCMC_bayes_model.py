import pandas as pd
import numpy as np
import warnings
import os
from scipy import stats
from scipy.stats import norm, gamma
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import arviz as az
from IPython.display import display

warnings.filterwarnings('ignore')

# TODO: add proper prior diistr
# TODO: change the way of gibbs sampling
# TODO: check histograms above fixes

class Prior:

    def __init__(self, y: pd.DataFrame, X: pd.DataFrame) -> None:
        self.y = y.to_numpy()
        self.X = X.to_numpy()
    
    def calc_a(self) -> np.float64 | np.ndarray:   ### TODO: change a priori
        a = np.mean(self.X, axis = 0)
        a = a.reshape(1, -1)
        
        return a.T
    
    def calc_C(self) -> np.float64 | np.ndarray:
        vars = np.var(self.X, axis = 0)
        C = np.diag(vars)
        C = C.T

        return C

class MCMCposterior:

    def __init__(self, 
        y: pd.DataFrame, X: pd.DataFrame,
        burnin: int, mcmc: int,
        a: int | float | np.ndarray, C: int | float | np.ndarray,
        n0: int | float | np.ndarray, s0: int | float | np.ndarray,
        beta_start: int | float | np.ndarray = None
        ) -> None:

        self.y = y.to_numpy().reshape(-1, 1)
        self.X = X.to_numpy()
        self.X_df = X
        self.independents = [_ for _ in X.columns]
        self.dependent = [_ for _ in y.name]
        self.burnin = burnin
        self.mcmc = mcmc
        self.a = a
        self.C = C
        self.n0 = n0
        self.s0 = s0
        self.beta_start = beta_start

    def calc_beta_start(self) -> None:

        def OLS(y: np.ndarray,  X: np.ndarray) -> np.ndarray:
            to_inv = X.T @ X
            X_inv = np.linalg.inv(to_inv)
            beta_hat = X_inv @ X.T @ y

            return beta_hat.T

        beta_start = OLS(self.y, self.X)
        beta_start = beta_start.reshape(1, -1)

        return beta_start


    def gibbs_sampling(self) -> None:

        def SSE(y: np.ndarray,  X: np.ndarray, beta: np.ndarray) -> float:
            residuals = y - (X @ beta.T).reshape(-1, 1)
            sse = residuals.T @ residuals

            return sse

        def calc_tau_param(T: int, sse: float, n0: float, s0: float) -> float:
            n_ = n0 + T
            s_ = s0 + sse

            return n_, s_
        
        def calc_beta_param(X: np.ndarray, C: np.ndarray, a: np.ndarray, tau: np.ndarray, beta: np.ndarray) -> np.ndarray:
            C_ = C + tau * X.T @ X
            C_inv = np.linalg.inv(C_)
            a_ = C_inv @ (C @ a + tau.item() * X.T @ X @ beta.T)

            return a_, C_
        
        def calc_beta() -> np.ndarray:
            a_, C_ = calc_beta_param(self.X, self.C, self.a, tau_matrix[-1], beta_hat)
            p_norm = np.random.uniform(0, 1, size = len(self.independents))
            z = norm.ppf(p_norm).reshape(-1, 1)
            C_inv = np.linalg.inv(C_)
            L = cholesky(C_inv) # TODO: random.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)

            norm_quantile_vec = a_ + L @ z

            return norm_quantile_vec
        
        def calc_tau() -> np.ndarray:
            sse = SSE(self.y, self.X, beta_matrix[-1])
            n_, s_ = calc_tau_param(T, sse, self.n0, self.s0)
            gamma_start_prob = np.random.uniform(0, 1)

            gamma_quantile_vec = gamma.ppf(gamma_start_prob, a = n_ / 2, scale = s_ / 2) # random.gamma(shape, scale=1.0, size=None)

            return gamma_quantile_vec

        if not self.beta_start:
            beta_matrix = self.calc_beta_start()
        else:
            beta_matrix = self.beta_start
        
        if beta_matrix.shape[0] > 1:
            beta_matrix = beta_matrix.T

        T = len(self.X)
        beta_hat = self.calc_beta_start()

        gamma_start_prob = np.random.uniform(0, 1)
        gamma_start_quantile = gamma.ppf(gamma_start_prob, a = self.n0 / 2, scale = self.s0 / 2)
        tau_matrix = np.array([[gamma_start_quantile]])

        total_iterations = self.burnin + self.mcmc
        for number in range(total_iterations):
            norm_quantile_vec = calc_beta()
            beta_matrix = np.vstack((beta_matrix, norm_quantile_vec.T))

            gamma_quantile_vec = calc_tau()
            tau_matrix = np.vstack((tau_matrix, gamma_quantile_vec))

        self.beta_matrix = beta_matrix
        self.tau_matrix = tau_matrix
    
    def preapare_data_for_plot(self, clear: bool = False, virable: str | list = None, all_virables: bool = False, include_tau: bool = False) -> pd.DataFrame:
        if include_tau:
            tau_data = pd.DataFrame(self.tau_matrix, columns = ['Precision'])   

        if all_virables:
            beta_data = pd.DataFrame(self.beta_matrix, columns = self.independents)

        if not all_virables:
            if not virable and not include_tau:
                raise Exception('Pass at least one virable name to get a traceplot.')
            elif virable:
                beta_data = pd.DataFrame(self.beta_matrix, columns = self.independents)
                beta_data = beta_data[virable]
        
        if clear:
            beta_data = beta_data.loc[self.burnin + 1:, :].reset_index(drop=True)

            if include_tau:
                tau_data = tau_data.loc[self.burnin + 1:, :].reset_index(drop=True)
        
        if include_tau:
            data = pd.concat([beta_data, tau_data], axis=1)
        else:
            data = beta_data
        
        return data
    
    def traceplot(self, clear: bool = False, virable: str | list = None, all_virables: bool = False, include_tau: bool = False) -> None:
        def plot_logic(dataframe: pd.DataFrame) -> list:
            traceplots = []
            traceplot_names = []
            
            for colname in dataframe.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_title(colname, color="#000000", loc="center", fontsize=8)
                ax.plot(dataframe[colname], color="lightcoral", linewidth=0.75, linestyle="-", alpha=0.9, label=None)

                mean_value = dataframe[colname].mean()
                ax.axhline(mean_value, color="black", linestyle="--", linewidth=0.8, alpha=0.8)

                y_pos = mean_value + 0.05 * (dataframe[colname].max() - dataframe[colname].min())
                ax.text(x=0.05 * len(dataframe), y=y_pos, s=f"Mean: {mean_value:.2f}", 
                        color="black", fontsize=8, fontweight="bold")

                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.4f}'))

                ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
                ax.yaxis.get_offset_text().set_visible(False)

                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", which="major", direction='in', labelsize=7)
                ax.tick_params(axis="y", which="major", direction='in', labelsize=7)
                ax.xaxis.set_ticks_position("none")
                ax.yaxis.set_ticks_position("none")
                ax.grid(True)
                ax.set_axisbelow(True)

                traceplots.append(fig)
                traceplot_names.append(colname)
                plt.close(fig)
            
            return traceplots, traceplot_names
        
        data = self.preapare_data_for_plot(clear, virable, all_virables, include_tau)
        self.traceplots, self.traceplot_names = plot_logic(data)
    
    def write_traceplots(self, path: str = None) -> None:
        if path == None:
            path = os.getcwd()

        for plot, name in zip(self.traceplots, self.traceplot_names):
            filepath = os.path.join(path, f'traceplot_{name}.png')
            plot.savefig(filepath, format='png')
            plt.close(plot)

    def histogram(self, clear: bool = False, virable: str | list = None, all_virables: bool = False, include_tau: bool = False, bins: int = 20) -> None:
        def plot_logic(dataframe: pd.DataFrame, bins: int = 20) -> list:
            histograms = []
            histogram_names = []

            for colname in dataframe.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_title(colname, color="#000000", loc="center", fontsize=8)

                counts, bin_edges = np.histogram(dataframe[colname], bins=bins)
                bin_width = bin_edges[1] - bin_edges[0]
                total_counts = sum(counts)
                probabilities = counts / total_counts

                ax.bar(bin_edges[:-1], probabilities, width=bin_width, alpha=1, color="#ff7595", edgecolor="#FFFFFF", linewidth=1.0)
                mean_value = dataframe[colname].mean()

                ax.axvline(mean_value, color="blue", linewidth=1.25, linestyle="--", label=f"Mean: {mean_value:.2f}")
                ax.text(mean_value, 0.02, f'Mean: {mean_value:.2f}', color='blue', fontsize=10, ha='center')

                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.4f}'))

                ax.set_xticklabels([])
                ax.set_xticks([])

                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", which="major", direction='in', labelsize=7)
                ax.tick_params(axis="y", which="major", direction='in', labelsize=7)
                ax.xaxis.set_ticks_position("none")
                ax.yaxis.set_ticks_position("none")
                ax.grid(True)
                ax.set_axisbelow(True)

                histogram_names.append(colname)
                histograms.append(fig)

                plt.close(fig)

            return histograms, histogram_names
        
        data = self.preapare_data_for_plot(clear, virable, all_virables, include_tau)
        self.histograms, self.histogram_names = plot_logic(data)
    
    def write_histograms(self, path: str = None) -> None:
        if path == None:
            path = os.getcwd()

        for plot, name in zip(self.histograms, self.histogram_names):
            filepath = os.path.join(path, f'histogram_{name}.png')
            plot.savefig(filepath, format='png')
            plt.close(plot)
    
    def acf(self, clear: bool = False, virable: str | list = None, all_virables: bool = False, include_tau: bool = False, lags: int = 30) -> None:
        def plot_logic(dataframe: pd.DataFrame, bins: int = 20) -> list:
            acfs = []
            acf_names = []

            for colname in dataframe.columns:
                title = f'ACF - {colname}'
                fig, ax = plt.subplots(figsize=(12, 6))

                plot = plot_acf(dataframe[colname], lags=lags, ax=ax)
                ax.set_title(title, fontsize=14)

                ax.set_ylim([-0.25, 1])

                acfs.append(plot)
                acf_names.append(colname)

                plt.close(fig)

            return acfs, acf_names

        data = self.preapare_data_for_plot(clear, virable, all_virables, include_tau)
        self.acfs, self.acf_names = plot_logic(data)

    def write_acfs(self, path: str = None) -> None:
        if path == None:
            path = os.getcwd()

        for plot, name in zip(self.acfs, self.acf_names):
            filepath = os.path.join(path, f'acf_{name}.png')
            plot.savefig(filepath, format='png')
            plt.close(plot)

    def describe_posterior(self, alpha: float = 0.05 , write: bool = False, path: str = None) -> None:
        def summary_statistics(dataframe: pd.DataFrame, alpha: float) -> pd.DataFrame:
            statistics = {}
            
            for col in dataframe.columns:
                col_data = dataframe[col]
                
                mean = np.mean(col_data)
                variance = np.var(col_data)
                std_dev = np.std(col_data)
                median = np.median(col_data)
                mode = stats.mode(col_data)[0]
                sem = stats.sem(col_data) # idk why not std??

                temp_array = np.array(dataframe[col])
                ci_lower, ci_upper = stats.norm.interval(confidence=(1-alpha), loc=mean, scale=sem)
                hdp_lower, hdp_upper = az.hdi(temp_array, hdi_prob=(1-alpha))
                
                statistics[col] = {
                    'Mean': mean,
                    'Variance': variance,
                    'Standard Deviation': std_dev,
                    'Median': median,
                    'Mode': mode,
                    'Credibility Interval Lower': ci_lower,
                    'Credibility Interval Upper': ci_upper,
                    'HDP Lower': hdp_lower,
                    'HDP upper': hdp_upper
                }
                for k, v in statistics[col].items():
                    statistics[col][k] = v
            return pd.DataFrame(statistics)

        data_df = self.preapare_data_for_plot(clear=True, all_virables=True, include_tau=True)
        statistics = summary_statistics(data_df, alpha)
        display(statistics)

        if write:
            if not path:
                path = os.getcwd()
            
            filename = path + '\\posterior_describe.xlsx'

            statistics.to_excel(filename)
    
    def describe_prior(self, write: bool = False, path: str = None) -> None:
        def summary_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
            statistics = {}
            
            for col in dataframe.columns:
                col_data = dataframe[col]
                
                mean = np.mean(col_data)
                variance = np.var(col_data)
                std_dev = np.std(col_data)
                median = np.median(col_data)
                mode = stats.mode(col_data)[0]
                
                statistics[col] = {
                    'Mean': mean,
                    'Variance': variance,
                    'Standard Deviation': std_dev,
                    'Median': median,
                    'Mode': mode
                }
                for k, v in statistics[col].items():
                    statistics[col][k] = v

            tau_mean = self.n0 / self.s0
            tau_var = self.n0 / (self.s0) ** 2
            tau_std = (tau_var) ** (1/2)
            tau_median = gamma.ppf(0.5, a = self.n0 / 2, scale = self.s0 / 2)
            
            if self.n0 < 1:
                tau_mode = 'Not exists'
            else:
                tau_mode = (self.n0 - 1) / self.s0

            statistics_tau = {
                    'Mean': tau_mean,
                    'Variance': tau_var,
                    'Standard Deviation': tau_std,
                    'Median': tau_median,
                    'Mode': tau_mode
                }

            statistics['Precision'] = statistics_tau

            return pd.DataFrame(statistics)

        data_df = self.X_df
        statistics = summary_statistics(data_df)
        display(statistics)

        if write:
            if not path:
                path = os.getcwd()
            
            filename = path + '\\posterior_describe.xlsx'

            statistics.to_excel(filename)

    def ergodic_mean(self, virable: str | list = None, all_virables: bool = False, include_tau: bool = False) -> None:
        def plot_logic(dataframe: pd.DataFrame) -> list:
            ergodic_means = []
            ergodic_names = []

            for colname in dataframe.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_title(f'Ergodic Mean - {colname}', color="#000000", loc="center", fontsize=10)

                data = dataframe[colname].values
                mean_ls = np.cumsum(data) / np.arange(1, len(data) + 1)

                ax.plot(mean_ls, color="blue", linewidth=1.5, label="Ergodic Mean")
                ax.axhline(y=mean_ls[-1], color="red", linestyle="--", linewidth=1.25, label=f"Final Mean: {mean_ls[-1]:.2f}")

                y_pos = mean_ls[-1] + 0.05 * (dataframe[colname].max() - dataframe[colname].min())
                ax.text(x=0.05 * len(data), y=y_pos, s=f"Mean: {mean_ls[-1]:.2f}", 
                        color="red", fontsize=8, fontweight="bold")

                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.4f}'))
                ax.yaxis.get_offset_text().set_visible(False)
                ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

                ax.set_xlabel("Iteration")
                ax.set_ylabel("Mean Value")
                ax.legend()
                ax.grid(True)
                ax.set_axisbelow(True)

                ergodic_names.append(colname)
                ergodic_means.append(fig)

                plt.close(fig)

            return ergodic_means, ergodic_names

        data = self.preapare_data_for_plot(True, virable, all_virables, include_tau)
        self.ergodic_means, self.egodic_mean_names = plot_logic(data)
    
    def write_ergodic_means(self, path: str = None) -> None:
        if path == None:
            path = os.getcwd()

        for plot, name in zip(self.ergodic_means, self.egodic_mean_names):
            filepath = os.path.join(path, f'ergodic_mean_{name}.png')
            plot.savefig(filepath, format='png')
            plt.close(plot)
    
    def ergodic_stand_mean(self, virable: str | list = None, all_virables: bool = False, include_tau: bool = False, all_in_one: bool = False) -> None:
        def plot_logic(dataframe: pd.DataFrame, all_in_one: bool = False) -> list:
            ergodic_means = []
            ergodic_names = []

            if all_in_one:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_title("Standardized Ergodic Means - All Variables", color="#000000", loc="center", fontsize=10)

            for colname in dataframe.columns:
                data = dataframe[colname].values
                mean_ls = np.cumsum(data) / np.arange(1, len(data) + 1)

                squared_diffs = (data - mean_ls) ** 2
                std_ls = np.sqrt(np.cumsum(squared_diffs) / np.arange(1, len(data) + 1))
                stand_mean_ls = (mean_ls - mean_ls[-1]) / std_ls

                if all_in_one:
                    ax.plot(stand_mean_ls, linewidth=1.5, label=colname)
                else:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.set_title(f'Standardized Ergodic Mean - {colname}', color="#000000", loc="center", fontsize=10)
                    ax.plot(stand_mean_ls, color="blue", linewidth=1.5, label="Standardized Ergodic Mean")
                    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.25, label="Final Mean (Standardized)")

                    y_pos = 0.05 * (np.nanmax(stand_mean_ls) - np.nanmin(stand_mean_ls))
                    ax.text(x=0.05 * len(data), y=y_pos, s=f"Final Mean: {mean_ls[-1]:.2f}", 
                            color="red", fontsize=8, fontweight="bold")

                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.4f}'))
                    ax.yaxis.get_offset_text().set_visible(False)
                    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Standardized Mean")
                    ax.legend()
                    ax.grid(True)
                    ax.set_axisbelow(True)

                    ergodic_names.append(colname)
                    ergodic_means.append(fig)
                    plt.close(fig)

            if all_in_one:
                ax.axhline(y=0, color="black", linestyle="--", linewidth=1.25, label="Final Mean (Standardized)")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Standardized Mean")
                ax.legend()
                ax.grid(True)
                ax.set_axisbelow(True)

                ergodic_means = [fig]
                ergodic_names = ["all_variables"]
                plt.close(fig)

            return ergodic_means, ergodic_names

        data = self.preapare_data_for_plot(True, virable, all_virables, include_tau)
        self.ergodic_stand_means, self.egodic_stand_mean_names = plot_logic(data, all_in_one)
    
    def write_ergodic_stand_means(self, path: str = None) -> None:
        if path == None:
            path = os.getcwd()

        for plot, name in zip(self.ergodic_stand_means, self.egodic_stand_mean_names):
            filepath = os.path.join(path, f'ergodic_stand_means_{name}.png')
            plot.savefig(filepath, format='png')
            plt.close(plot)