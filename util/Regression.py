#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.optimize import curve_fit
import statistics
import math
import statsmodels.stats.stattools as sss
import statsmodels.stats.outliers_influence as sso
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
# 設定圖形大小; DPI越大圖越大
plt.rcParams["figure.dpi"] = 100
plt.rcParams['axes.unicode_minus'] = False
used = True


def data(xlsx):
    if (used):
        xlsx = "data used/" + xlsx
    return xlsx


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def Emphasize(string, emphasis=''):
    if (emphasis == ''):
        return color.BOLD + string + color.END
    else:
        string1, string2 = string.split(emphasis)
        return color.BOLD + string1 + color.RED + emphasis + color.END + color.BOLD + string2 + color.END


def durbin_table(tail, alpha, T, K):
    if (tail == 'two'):
        alpha = alpha / 2
    durbin = pd.ExcelFile(data('durbin-watson.xlsx'))
    tmp = durbin.parse("5%")
    tmp.columns = tmp.iloc[1]
    tmp = tmp.drop([0, 1])
    durbin5 = tmp.iloc[:, 0:4]
    for i in range(1, 20):
        durbin5 = pd.concat(
            [durbin5, tmp.iloc[:, 4*i:4*i+4].dropna()], ignore_index=True)
    tmp = durbin.parse("2.5%")
    tmp.columns = tmp.iloc[1]
    tmp = tmp.drop([0, 1])
    durbin2_5 = tmp.iloc[:, 0:4]
    for i in range(19):
        durbin2_5 = pd.concat(
            [durbin2_5, tmp.iloc[:, 4*i:4*i+4]], ignore_index=True)
    tmp = durbin.parse("1%")
    tmp.columns = tmp.iloc[1]
    tmp = tmp.drop([0, 1])
    durbin1 = tmp.iloc[:, 0:4]
    for i in range(19):
        durbin1 = pd.concat(
            [durbin1, tmp.iloc[:, 4*i:4*i+4]], ignore_index=True)
    if (alpha == 0.01):
        return ([durbin1.loc[durbin1['T'] == T].loc[durbin1['K'] == K, 'dL'].sum(), durbin1.loc[durbin1['T'] == T].loc[durbin1['K'] == K, 'dU'].sum()])
    elif (alpha == 0.05):
        return ([durbin5.loc[durbin5['T'] == T].loc[durbin5['K'] == K, 'dL'].sum(), durbin5.loc[durbin5['T'] == T].loc[durbin5['K'] == K, 'dU'].sum()])
    elif (alpha == 0.025):
        return ([durbin2_5.loc[durbin2_5['T'] == T].loc[durbin2_5['K'] == K, 'dL'].sum(), durbin2_5.loc[durbin2_5['T'] == T].loc[durbin2_5['K'] == K, 'dU'].sum()])


class Test1:
    def __init__(self, test, tail, hypothesis):
        self.test = test
        self.tail = tail
        self.hypothesis = hypothesis

    def standardize(self, mean, std, size):
        # 檢定假設，std已知/未知，df.shape[0]，df.mean()[0]
        if (self.test == 'chi'):
            return round((size - 1) * (std / self.hypothesis) ** 2, 4)
        if (std == 0):
            if (mean >= 1):
                mean = mean / size
            std = (self.hypothesis * (1 - self.hypothesis)) ** 0.5
        return round((mean - self.hypothesis) / (std / size ** 0.5), 4)

    def confidence_interval(self, mean, std, size, target, adjustment=1, unit="請填入unit", print_mean=False, wilson=None):
        # 樣本均數，已知/未知，df.shape[0]，0.95
        if (self.test == 'norm'):
            a = stats.norm()
        elif (self.test == 't'):
            a = stats.t(df=size - 1)
        elif (self.test == 'chi'):
            a = stats.chi2(df=size - 1)
            lcl = ((size - 1) * (std**2)) / \
                a.ppf((1 + target) / 2) * adjustment
            ucl = ((size - 1) * (std**2)) / \
                a.ppf((1 - target) / 2) * adjustment
            # if(print_mean):
            # print std
            print(
                f"{target * 100:.0f}% confidence interval：{lcl:.4f} to {ucl:.4f}{unit}2")
            return
        if (std == 0):
            if (wilson):
                mean += 2
                size += 4
            mean = mean / size
            std = (mean*(1-mean))**0.5
            if not (wilson) and not (size * mean > 5 and size * (1 - mean) > 5):
                return 0
        z = a.ppf((1 + target) / 2)
        lcl = (mean - ((z * std) / (size**0.5))) * adjustment
        ucl = (mean + ((z * std) / (size**0.5))) * adjustment
        if (print_mean):
            if (unit == "p"):
                print(f"{unit} = {mean * adjustment:.4f}", end=", ")
            else:
                print(f"mean：{mean * adjustment:.4f} {unit}", end=", ")
        if (unit == 'p'):
            unit = ''
        print(f"{target * 100:.0f}% confidence interval：{lcl:.4f} to {ucl:.4f} {unit}")

    def critical_value(self, alpha, dof=0, mean=0, std=1, size=1):
        # 0.5，df.shape[0] - 1，檢定假設，std已知/未知，df.shape[0]
        if (self.test == 'norm'):
            a = stats.norm()
        elif (self.test == 't'):
            a = stats.t(df=dof)
        elif (self.test == 'chi'):
            a = stats.chi2(df=dof)
        elif (self.test == 'f'):
            a = stats.f(dfn=dof[0], dfd=dof[1])
        if (self.tail == 'left'):
            return round((std / size ** 0.5) * a.ppf(alpha) + mean, 4)
        elif (self.tail == 'right'):
            return round((std / size ** 0.5) * a.isf(alpha) + mean, 4)
        elif (self.tail == 'two'):
            return [round((std / size ** 0.5) * a.ppf(alpha / 2) + mean, 4), round((std / size ** 0.5) * a.isf(alpha / 2) + mean, 4)]

    def p_value(self, target, dof=0, judge=0):
        # 標準化完，'right'，檢定，df.shape[0] - 1，df.shape[0] - 3
        target = target - judge
        if (self.test == 'norm'):
            a = stats.norm()
        elif (self.test == 't'):
            a = stats.t(df=dof)
        elif (self.test == 'chi'):
            a = stats.chi2(df=dof)
        elif (self.test == 'f'):
            a = stats.f(dfn=dof[0], dfd=dof[1])
        if (self.tail == 'left'):
            return round(a.cdf(target), 4)
        elif (self.tail == 'right'):
            return round(a.sf(target), 4)
        elif (self.tail == 'two'):
            if (target < 0):
                return round(a.cdf(target + judge) * 2, 4)
            else:
                return round(a.sf(target + judge) * 2, 4)


class Chi2():
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def critical_value(self, mean=0, std=1, size=1):
        a = stats.chi2(df=self.dof)
        return round((std / size ** 0.5) * a.isf(self.alpha) + mean, 4)


class GoFT(Chi2):
    def __init__(self, hypothesis, df, target_title='frequency'):
        if (df.shape[1] == 1):
            df = pd.DataFrame(df.value_counts(
                sort=False), columns=['frequency'])
        self.df = df
        self.size = self.df[target_title].size
        self.hypothesis = hypothesis * self.df[target_title].sum()
        self.dof = self.size - 1
        self.observation = df[target_title].values

    def rule_of_five(self):
        a = len(self.hypothesis)
        for i in range(-1, -len(self.hypothesis)-1, -1):
            if (self.hypothesis[i] < 5):
                #                 print(i)
                if (-i == len(self.hypothesis)):
                    self.hypothesis[i + 1] = self.hypothesis[i +
                                                             1] + self.hypothesis[i]
                    self.observation[i + 1] = self.observation[i +
                                                               1] + self.observation[i]
                else:
                    self.hypothesis[i - 1] = self.hypothesis[i -
                                                             1] + self.hypothesis[i]
                    self.observation[i - 1] = self.observation[i -
                                                               1] + self.observation[i]
                self.hypothesis = np.delete(self.hypothesis, i)
                self.observation = np.delete(self.observation, i)
#                 print(self.hypothesis)
                return self.rule_of_five()
                break
        if not (self.size == len(self.hypothesis)):
            print(Emphasize("已被修改，Data Meet Rules of Five"))
        else:
            print(Emphasize("Data Meet Rules of Five"))

    def goodness_of_fit_test(self, alpha):
        self.rule_of_five()
        self.alpha = alpha
        self.Chi, self.P = stats.chisquare(self.observation, self.hypothesis)
        print("Chi-stat = %0.4f, p-value = %0.4f" % (self.Chi, self.P))
        self.Chi_cv = self.critical_value()
        print("Chi critical value = %0.4f (degree of freedom = %d)" %
              (self.Chi_cv, self.dof))


class GoFT_N(GoFT):
    def __init__(self, observation=None):
        self.observation = observation

    def grouping(self, data, n_group, upperbound=4, lowerbound=-4, by='probability', constant='value'):
        if (constant == 'probability'):
            m = np.mean(data)
            s = np.std(data)
            probability_bins = np.full(n_group, 1 / n_group)
            z_bins = np.zeros((n_group + 1))
            z_bins[0] = lowerbound
            z_bins[n_group] = upperbound
            for j in range(1, n_group):  # 反標準化
                z_bins[j] = m + stats.norm.ppf(j * probability_bins[j]) * s
            actual_counts, actual_bins = np.histogram(data, bins=z_bins)
            self.hypothesis = actual_counts.sum() * probability_bins
            self.size = len(self.hypothesis)
            self.dof = self.size - 3
            if (self.observation == None):
                self.observation = actual_counts

    def goodness_of_fit_test(self, alpha):
        self.rule_of_five()
        self.alpha = alpha
        self.Chi, self.P = stats.chisquare(
            self.observation, self.hypothesis, ddof=2)
        print("Chi-stat = %0.4f, p-value = %0.4f" % (self.Chi, self.P))
        self.Chi_cv = self.critical_value()
        print("Chi critical value = %0.4f (degree of freedom = %d)\n" %
              (self.Chi_cv, self.dof))
        if (self.P > 0.05):
            print(color.BOLD + "Since p-value %.4f > 0.05, there isn't enough evidence to reject the null hypothesis：we conclude that the normality assumption" % self.P
                  + color.RED + " is not violated." + color.END)
        else:
            print(color.BOLD + "Since p-value %.4f < 0.05, there is enough evidence to reject the null hypothesis：we conclude that the normality assumption" % self.P
                  + color.RED + " is violated." + color.END)


class IT(Chi2):
    # 表格樣式：主要兩欄，或第一因子在上，第二因子是資料，右尾，nominal data
    def __init__(self, df, target_title=[]):
        if not (df.shape[1] == 2):
            df = df.melt(var_name=target_title[0], value_name=target_title[1]).dropna(
            ).reset_index().drop(['index'], axis=1)
        self.df = df
        self.cont1 = pd.crosstab(
            self.df[target_title[0]], self.df[target_title[1]])

    def rule_of_five(self):
        if ((self.cont1 < 5).sum().sum() == 0):
            print("Data Meet Rules of Five")
        else:
            print("請修改")

    def independence_test(self, alpha):
        self.alpha = alpha
        self.rule_of_five()
        self.Chi, self.P, self.dof, ex = stats.chi2_contingency(
            self.cont1, correction=False)
        print("Chi-stat = %0.4f, p-value = %0.4f" % (self.Chi, self.P))
        self.Chi_cv = self.critical_value()
        print("Chi critical value = %0.4f (defree of freedom = %d)" %
              (self.Chi_cv, self.dof))
        print("Expected Frequency:")
        print(ex)


class Regression(Test1):
    def __init__(self, df, target_title={}, alpha=0.05):
        self.df = df
        self.y = self.df[target_title['y']].dropna()
        self.x = self.df[target_title['x']].dropna()
        self.target_title = target_title
        self.result = smf.ols('%s~ %s' % (
            self.target_title['y'], self.target_title['x']), data=self.df).fit()
        self.residual = sso.summary_table(self.result, alpha=0.05)[1][:, 10]
        self.b1 = self.result.params[1]
        self.b0 = self.result.params[0]
        self.r2 = round(self.result.rsquared, 4)
        self.r = round(
            self.result.params[1] / abs(self.result.params[1]) * (self.result.rsquared ** 0.5), 4)
        self.x_bar = self.x.mean()
        self.sample_size = self.df.shape[0]
        self.Sxx = np.cov(self.y, self.x)[1, 1]
        self.standard_error = self.result.mse_resid ** 0.5

    def scatter(self, line, conclusion=True):
        if (line == False):
            plt.scatter(self.x, y=self.y, color='r')
        else:
            _ = sns.regplot(
                x=self.target_title['x'], y=self.target_title['y'], data=self.df, color='b', ci=None)
            plt.xlim(self.x.min() * 0.9, self.x.max() * 1.05)
        plt.title('Scatter Plot for %s and %s' %
                  (self.target_title['y'], self.target_title['x']))
        plt.xlabel(self.target_title['x'])
        plt.ylabel(self.target_title['y'])
        plt.show()
        if (conclusion):
            print(color.BOLD +
                  "According to the scatter plot, we can see that there is linear relationship between %s and %s." % (self.target_title['x'], self.target_title['y']))
            print(
                "Thus, we can apply Simple linear regression with OLS." + color.END + '\n')

    def line(self, plot=True, liner=False, conclusion=True):
        if (plot):
            self.scatter(liner, conclusion)
        print(color.BOLD + "Estimated model: y = %0.4f + %0.4f x\n" %
              (self.b0, self.b1) + color.END)

    def beta_test(self, tail, alpha=0.05):
        self.tail = tail
        self.test = 't'
        T = self.result.tvalues[self.target_title['x']]
        self.dof = self.result.df_resid
        tcv = self.critical_value(alpha, self.result.df_resid)
        pvalue = self.p_value(T, self.result.df_resid)
#             print(self.result.pvalues[self.target_title['x']])
        print("T-stat = %0.4f, p-value = %0.4f" % (T, pvalue))
        print("T critical value %s tail =" % (tail), tcv,
              "(degree of freedom = %d)" % self.result.df_resid)
        del self.tail
        del self.test

    def standard_error_estimate(self):
        print("Standard Error =", round(self.standard_error, 4))
        print("Mean of y =", round(self.y.mean(), 4))
        print("Std of y =", round(self.y.std(), 4))

    def c_of_c_test(self, tail, alpha=0.05):
        n = self.df.shape[0]
        self.tail = tail
        self.test = 't'
        T = self.r * ((n - 2) / (1 - self.r ** 2)) ** 0.5
        tcv = self.critical_value(alpha, self.result.df_resid)
        pvalue = self.p_value(T, self.result.df_resid)
        print("Coefficient of Correlation =", self.r)
        print("\nT-stat = %0.4f, p-value = %0.4f" % (T, pvalue))
        print(f"T critical value one tail =", tcv,
              "(defree of freedom = %d)" % self.result.df_resid)
        del self.tail
        del self.test

    def descriptive_measurement(self):
        print("Coefficient of Determination =", self.r2, end='')
        print(" (remaining %0.4f)" % (1 - self.r2))

    def interval(self, type_, target, alpha=0.95, unit='請輸入單位'):
        new_target = np.array([1, target])
        X2 = sm.add_constant(self.x)
        result_reg = sm.OLS(self.y, X2).fit()
        y_head = np.dot(result_reg.params, new_target)
        (t_minus, t_plus) = stats.t.interval(
            alpha=alpha, df=result_reg.df_resid)
        if (type_ == 'Confidence'):
            core = (1 / self.sample_size + (target - self.x_bar) **
                    2 / (self.sample_size - 1) / self.Sxx) ** 0.5
        elif (type_ == 'Prediction'):
            core = (1 + 1 / self.sample_size + (target - self.x_bar)
                    ** 2 / (self.sample_size - 1) / self.Sxx) ** 0.5
        lcl = y_head + t_minus * (result_reg.mse_resid ** 0.5) * core
        ucl = y_head + t_plus * (result_reg.mse_resid ** 0.5) * core
        print(
            f"{alpha * 100:.0f}% {type_} interval：{lcl:.4f} to {ucl:.4f} {unit} at mean of x = {target}")

    def residual_ypredict(self, set_ylim=[None, None], outlier=False, influential_observation=False, standardized=True, conclusion=True):
        self.y_pre = sso.summary_table(self.result, alpha=0.05)[1][:, 2]
        if not (standardized):
            plt.plot(self.y_pre, self.y-self.y_pre, 'o', color='#7209b7')
        else:
            plt.plot(self.y_pre, self.residual, 'o', color='#7209b7')
            plt.ylim(set_ylim[0], set_ylim[1])
            plt.axhline(y=2, color='red')
            plt.axhline(y=-2, color='red')
        plt.axhline(y=0, color='blue')
        plt.title('Plot of Residuals vs Predicted')
        plt.xlabel("Predicted " + self.target_title['y'])
        plt.ylabel("Residuals")
        plt.show()
        if (outlier):
            self.outlier()
        if (influential_observation):
            print('\n')
            self.influential_observation()
        if (influential_observation == 'Cook'):
            print('\n')
            self.influential_observation('Cook')
        if (conclusion):
            print("\n")
            print(Emphasize("According to the figure, there isn't enough evidence to reject the null hypothesis：We can assume that the variation is constant and the mean is around 0."))
        else:
            print("\n")
            print(Emphasize("According to the figure, there is enough evidence to reject the null hypothesis：We can\'t assume that the variation is constant and the mean is around 0.", "can\'t"))

    def outlier(self):
        df = pd.DataFrame(self.residual, columns=['Standardized Residual'])
        filter = (df['Standardized Residual'] < -
                  2) | (df['Standardized Residual'] > 2)
        if (len(df['Standardized Residual'].loc[filter]) == 0):
            print(Emphasize("No Outliers"))
        else:
            print("Outliers by Standardized Residual =\n")
            print(df['Standardized Residual'].loc[filter])

    def influential_observation(self, type_=False):
        h = 1 / self.sample_size + \
            (self.x - self.x_bar) ** 2 / (self.sample_size - 1) / self.Sxx
        df = h.to_frame('hi')
        filter = (df['hi'] > 6 / self.sample_size)
        if (len(df['hi'].loc[filter]) == 0):
            print(Emphasize("No Influential Observations"))
        else:
            print("Influential Observations by hi =\n")
            print(df['hi'].loc[filter])

    def hist(self, conclusion=True):
        counts, bins, patches = plt.hist(
            self.residual, bins=6, density=False, facecolor='#4cc9f0', alpha=0.75)
        plt.xlabel('Standardized Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Standardized Residuals')
        plt.grid(True)
        bin_centers = [np.mean(k) for k in zip(bins[:-1], bins[1:])]
        plt.show()
        if (conclusion):
            print(Emphasize("The histogram looks like a bell-shaped distribution."))
        else:
            print(Emphasize(
                "The histogram doesn't look like a bell-shaped distribution.", "doesn't"))

    def qqplot(self, conclusion=True):
        fig = sm.qqplot(self.residual, stats.norm, fit=True, line='45')
        plt.show()
        if (conclusion):
            print(Emphasize(
                "Q-Q Plot does show that all data points are close to the 45 degree line."))
        else:
            print(Emphasize(
                "Q-Q Plot does show that not all data points are close to the 45 degree line.", "not"))

    def Chi_Square_for_Normality(self, group):
        This = GoFT_N()
        This.grouping(self.residual, group, by='value', constant='probability')
        This.goodness_of_fit_test(0.05)

    def shapiro(self):
        stat, p = stats.shapiro(self.residual)
        print('Statistics=%.4f, p=%.4f\n' % (stat, p))
        if (p > 0.05):
            print(color.BOLD + "Since p-value %.4f > 0.05, there isn't enough evidence to reject the null hypothesis：we conclude that the normality assumption" % p
                  + color.RED + " is not violated." + color.END)
        else:
            print(color.BOLD + "Since p-value %.4f < 0.05, there is enough evidence to reject the null hypothesis：we conclude that the normality assumption" % p
                  + color.RED + " is violated." + color.END)

    def runs_test(self, lb=0, ub=0):
        median = statistics.median(self.residual)
        runs, n1, n2 = 1, 0, 0
        if (self.residual[0] >= median):
            n1 += 1
        else:
            n2 += 1
        for i in range(1, len(self.residual)):
            if (self.residual[i] >= median and self.residual[i-1] < median) or (self.residual[i] < median and self.residual[i-1] >= median):
                runs += 1
            if (self.residual[i]) >= median:
                n1 += 1
            else:
                n2 += 1
        if not (n1 > 20 and n2 > 20):
            print('n1 =', n1, ', n2 =', n2, ', runs =', runs)
            print(Emphasize('Check Table'))
            img = mpimg.imread('runs_test.jpg')
            imgplot = plt.imshow(img)
            plt.xticks()
            plt.show()
            if not (lb == ub):
                if (runs > lb and runs < ub):
                    print(color.BOLD + "From the Runs Table, since %d < Runs = %d < %d, there isn't enough evidence to reject the null hypothesis：we conclude that the randomness assumption" % (lb, runs, ub)
                          + color.RED + " is not violated." + color.END)
                else:
                    print(color.BOLD + "From the Runs Table, since Runs = %d is out of the range [%d, %d] , there is enough evidence to reject the null hypothesis：we conclude that the randomness assumption" % (runs, lb, ub)
                          + color.RED + " is violated." + color.END)
            return
        exp = ((2 * n1 * n2) / (n1 + n2)) + 1
        std = ((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
               (((n1 + n2) ** 2)*(n1 + n2 - 1))) ** 0.5
        Z = (runs - exp) / std
        pvalue = stats.norm.sf(abs(Z)) * 2
        print("Z-stat = %0.4f, p-value = %0.4f\n" % (Z, pvalue))
        if (pvalue > 0.05):
            print(color.BOLD + "Since p-value %.4f > 0.05, there isn't enough evidence to reject the null hypothesis：we conclude that the randomness assumption" % pvalue
                  + color.RED + " is not violated." + color.END)
        else:
            print(color.BOLD + "Since p-value %.4f < 0.05, there is enough evidence to reject the null hypothesis：we conclude that the randomness assumption" % pvalue
                  + color.RED + " is violated." + color.END)


class Regression_M(Regression):
    def __init__(self, df, target_title={}, alpha=0.05):
        self.df = df
        self.sample_size = self.df.shape[0]
        self.y = self.df[target_title['y']].dropna()
        self.x = [[1.0]]
        self.x_name = ['']
        self.x_df = pd.DataFrame([1.0] * self.sample_size, columns={'const'})
        for i in target_title:
            if (i == 'y'):
                continue
            self.x_name.append(target_title[i])
            self.x.append(self.df[target_title[i]].dropna())
            self.x_df[target_title[i]] = self.x[-1]
        self.independent_variable = len(self.x) - 1  # x數
        self.target_title = target_title
        self.result = sm.OLS(self.y, self.x_df).fit()
        self.residual = sso.summary_table(self.result, alpha=0.05)[1][:, 10]
        self.b = []
        for i, k in enumerate(self.result.params):
            self.b.append(self.result.params[i])
        self.r2 = round(self.result.rsquared, 4)
        self.adjusted_r2 = round(self.result.rsquared_adj, 4)
        self.standard_error = self.result.mse_resid ** 0.5
        self.mul_C = False
        self.mul_T = False

    def scatter(self, line, fontsize=6, spot=1, conclusion=True):
        for i in range(1, self.independent_variable + 1):
            ax = plt.subplot((math.ceil(self.independent_variable / 3)), 3, i)
            if (line == False):
                fig = plt.scatter(self.x[i], y=self.y, color='r', s=spot)
            else:
                fig = sns.regplot(
                    x=self.x[i], y=self.y, data=self.df, color='b', ci=None, scatter_kws={'s': spot})
                plt.xlim(self.x[i].min() * 0.95, self.x[i].max() * 1.05)
            plt.title('Scatter Plot for %s and %s' % (
                self.target_title['y'], self.x_name[i]), fontsize=fontsize * 1.5)
            plt.xlabel(self.x_name[i], fontsize=fontsize * 1.5)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.ylabel(self.target_title['y'], fontsize=fontsize * 1.5)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                            top=0.9, wspace=0.4, hspace=0.4)
        plt.show()
        if (conclusion):
            print(color.BOLD +
                  "According to the scatter plot, all independet variables seem to have linear relationship with %s." % (self.target_title['y']) + color.END + '\n')

    def line(self, plot=True, liner=True, fontsize=6, spot=1, conclusion=True):
        if (plot):
            self.scatter(liner, fontsize, spot, conclusion)
        # print(color.BOLD + "Estimated model: %s = %0.4f" %
            #   (self.target_title['y'], self.b[0]), end='')
        # for i, k in enumerate(self.x_name[1:], 1):
        #     print(" + %0.4f %s" % (self.b[i], k), end='')
        # print(color.END + '\n')

    def descriptive_measurement(self, adjusted=False, interpret=True):
        if not (adjusted):
            print("Coefficient of Determination =", self.r2, end='')
            print(" (remaining %0.4f)" % (1 - self.r2))
            if (interpret):
                print("\n" + color.BOLD
                      + "This statistic tells us that %0.2f%% of the variation in %s is explained by this regression line of the independent variables. The remaining %.2f%% is unexplained" % (
                          self.r2 * 100, self.target_title['y'], (1 - self.r2) * 100)
                      + color.END + "\n")
        else:
            print("Adjusted Coefficient of Determination =",
                  self.adjusted_r2, end='')
            print(" (differ in %0.4f)" %
                  ((self.r2 - self.adjusted_r2) / self.r2))
            if (interpret):
                if ((self.r2 - self.adjusted_r2) / self.r2 < 0.06):
                    print("\n" + color.BOLD
                          + "The adjusted coefficient of determination is %0.2f%%, which is close to %0.2f%% (< 0.06), indicating that the model has not problem of over-fitting." % (
                              self.adjusted_r2 * 100, self.r2 * 100)
                          + color.END)
                else:
                    print("\n" + color.BOLD
                          + "The adjusted coefficient of determination is %0.2f%%, which is not close to %0.2f%% (> 0.06), indicating that the model" % (
                              self.adjusted_r2 * 100, self.r2 * 100) + color.RED + " may have" + color.END
                          + color.BOLD + " some problem of over-fitting." + color.END)

    def anova_test(self, alpha=0.05, return_=False):
        self.tail = 'right'
        self.test = 'f'
        F = self.result.fvalue
        self.dof_r = self.result.df_resid
        self.dof_m = self.result.df_model
        fcv = self.critical_value(alpha, [self.dof_m, self.dof_r])
        pvalue = self.p_value(F, [self.dof_m, self.dof_r])
#             print(self.result.pvalues[self.target_title['x']])
        print("F-stat = %0.4f, p-value = %0.4f" % (F, pvalue))
        print(f"F critical value one tail =", fcv,
              "(degree of freedom = %d, %d)\n" % (self.dof_m, self.dof_r))
        del self.tail
        del self.test
        if (F > fcv):
            print(color.BOLD + "Since %.4f > %.4f, there is enough evidence to reject the null hypothesis：We can infer that at least one of the Beta_i is not equal to zero," %
                  (F, fcv) + Emphasize(" which means this regression model is valid.", "is valid"))
            print(color.BOLD + "We can also confirm this since p-value %.4f < 0.05" %
                  pvalue + color.END)
            if (return_):
                return True
        else:
            print(color.BOLD + "Since %.4f < %.4f, there isn't enough evidence to reject the null hypothesis" % (F, fcv) + Emphasize(
                "：We can't infer that at least one of the Beta_i is not equal to zero,", "can't") + Emphasize(" which means this regression model is not valid.", "is not valid"))
            print(color.BOLD + "We can also confirm this since p-value %.4f > 0.05" %
                  pvalue + color.END)
            if (return_):
                return False

    def beta_test(self, tail, alpha=0.05, return_=False):
        significance = False
        for i in range(1, self.independent_variable + 1):
            if (i == 1):
                print('b'+str(i)+' (%s):' % (self.x_name[i]))
            else:
                print('\nb'+str(i)+' (%s):' % (self.x_name[i]))
            self.tail = tail
            self.test = 't'
            T = self.result.tvalues[i]
            self.dof = self.result.df_resid
            tcv = self.critical_value(alpha, self.result.df_resid)
            pvalue = self.result.pvalues[i]
            print("T-stat = %0.4f, p-value = %0.4f" % (T, pvalue))
            print("T critical value %s tail =" % (tail), tcv,
                  "(degree of freedom = %d)" % self.result.df_resid)
            if (pvalue < 0.05):
                print(color.BOLD + "Since the p value %0.4f < 0.05, there is enough evidence to reject the null hypothesis：" % pvalue +
                      Emphasize("We can infer that it is linearly related to ", "can") + color.BOLD + self.target_title['y'] + '.' + color.END)
                significance = True
            else:
                print(color.BOLD + "Since the p value %0.4f > 0.05, there is enough evidence to reject the null hypothesis：" % pvalue +
                      Emphasize("We can\'t infer that it is linearly related to ", "can\'t") + color.BOLD + self.target_title['y'] + '.' + color.END)
            del self.tail
            del self.test
        if (return_):
            return significance

    def interval(self, type_, target, alpha=0.95, unit='請輸入單位'):
        new_target = np.array([1]+target)
        X = self.x_df.values
        y_head = np.dot(self.result.params, new_target)
        (t_minus, t_plus) = stats.t.interval(
            alpha=alpha, df=self.result.df_resid)
        if (type_ == 'Confidence'):
            core = (self.result.mse_resid * np.matmul(new_target,
                    np.linalg.solve(np.matmul(X.T, X), new_target))) ** 0.5
        elif (type_ == 'Prediction'):
            core = (self.result.mse_resid * (1 + np.matmul(new_target,
                    np.linalg.solve(np.matmul(X.T, X), new_target)))) ** 0.5
        lcl = y_head + t_minus * core
        ucl = y_head + t_plus * core
        print(
            f"{alpha * 100:.0f}% {type_} interval：{lcl:.4f} to {ucl:.4f} {unit} at mean of x = {target}")

    def c_of_c_table(self, added_item):
        conflict = False
        coef = self.result.params.to_frame().rename(
            columns={0: self.target_title['y']})
        self.corr = pd.concat([self.y, self.x_df.iloc[:, 1:]], axis=1).corr()
        _ = sns.heatmap(self.corr, annot=True)
        plt.show()
        for i in self.x_name[1:]:
            if (i == added_item):
                continue
            a = coef[self.target_title['y']][i] / \
                abs(coef[self.target_title['y']][i])
            b = self.corr[self.target_title['y']][i] / \
                abs(self.corr[self.target_title['y']][i])
            if not (a == b):
                print(color.BOLD + color.RED +
                      '%s: coefficients conflict' % i + color.END)
                conflict = True
            else:
                print('%s: no conflicts' % i)
        return conflict

    def multicollinearity(self, type_, added_item='', conclusion=False):
        if (type_ == 'coefficient'):
            conflict = self.c_of_c_table(added_item)
            if not (conflict):
                print("\n" + Emphasize("From the scatter plots and the coefficient tables, the signs of coefficients of each independent variables do not contradict.", "do not contradict"))
                self.mul_C = True
            else:
                print("\n" + Emphasize("From the scatter plots and the coefficient tables, the signs of coefficients of some independent variables do contradict.", "do contradict"))
        elif (type_ == 'F&T'):
            F = self.anova_test(True)
            print('\n')
            T = self.beta_test('two', True)
            print('\n')
            if (F == T):
                print(color.BOLD + Emphasize("Based on the above two results, these two tests do not contradict.",
                      "do not contradict") + color.END)
                self.mul_T = True
            else:
                print(color.BOLD + Emphasize("Based on the above two results, these two tests do contradict.",
                      "do contradict") + color.END)
        if (conclusion):
            print('\n')
            if (self.mul_C and self.mul_T):
                print(
                    color.BOLD + "From these two checks, we conclude that the model does not have multicollinearity.")
            else:
                print(color.BOLD + Emphasize(
                    "From these two checks, we conclude that multicollinearity exists.", "exists"))

    def residual_time(self, set_ylim=[None, None]):
        self.time = sso.summary_table(self.result, alpha=0.05)[1][:, 0]
        plt.plot(self.time, self.residual, 'o', color='#7209b7')
        plt.ylim(set_ylim[0], set_ylim[1])
        plt.axhline(y=0, color='blue')
        plt.axhline(y=2, color='red')
        plt.axhline(y=-2, color='red')
        plt.title('Plot of Residuals vs Time')
        plt.xlabel("Observation No.")
        plt.ylabel("Residuals")
        plt.show()

    def durbin_watson(self, alpha, tail):
        D = sss.durbin_watson(self.residual)
        print('n=', self.sample_size, ', k=', len(self.b) - 1)
        print("D-stat = %0.4f\n" % (D))
        if (self.sample_size > 100):
            return
        DL, DU = tuple(durbin_table(
            tail, alpha, self.sample_size, len(self.b)))
        print(color.BOLD + "From the Durbin-Watson table, the DL and DU are %.4f and %.4f" %
              (DL, DU) + color.END)
        if (DU < D < 2):
            print(color.BOLD + "Since DU = %.4f < D = %.4f < 2, there isn't enough evidence to reject the null hypothesis：" %
                  (DU, D) + Emphasize("We can't infer that first order correlation exists.", "can't") + color.END)
        elif (2 < D < 4 - DU):
            print(color.BOLD + "Since 2 < D = %.4f < 4 - DU = %.4f, there isn't enough evidence to reject the null hypothesis：" %
                  (D, 4 - DU) + Emphasize("We can't infer that first order correlation exists.", "can't") + color.END)
        elif (DL < D < DU):
            print(color.BOLD + "Since DL = %.4f < D = %.4f < DU = %.4f, the test is inconclusive" %
                  (DL, D, DU) + color.END)
        elif (4 - DU < D < 4 - DL):
            print(color.BOLD + "Since 4 - DU = %.4f < D = %.4f < 4 - DL = %.4f, the test is inconclusive" %
                  (4 - DU, D, 4 - DL) + color.END)
        elif (D < DL):
            print(color.BOLD + "Since D = %.4f < DL = %.4f, there is enough evidence to reject the null hypothesis：" %
                  (D, DL) + Emphasize("We can infer that first order correlation exists.", "can") + color.END)
        elif (D > 4-DL):
            print(color.BOLD + "Since D = %.4f > 4 - DL = %.4f, there is enough evidence to reject the null hypothesis：" %
                  (D, 4 - DL) + Emphasize("We can infer that first order correlation exists.", "can") + color.END)

    def influential_observation(self, type_=False):
        X = self.x_df.values
        H = np.matmul(X, np.linalg.solve(np.matmul(X.T, X), X.T))
        df = pd.DataFrame(np.diagonal(H)).rename(columns={0: 'hii'})
        k = self.result.df_model
        if not (type_):
            n = len(df['hii'])
            h_level = 3 * (k+1) / n
            filter = (df['hii'] > h_level)
            if (len(df['hii'].loc[filter]) == 0):
                print(Emphasize("No Influential Observations"))
            else:
                print("Influential Observations by hi =\n")
                print(df['hii'].loc[filter])
        else:
            s2_e = self.result.mse_resid
            k = self.result.df_model
            y_a = sso.summary_table(self.result, alpha=0.05)[1][:, 1]
            y_f = sso.summary_table(self.result, alpha=0.05)[1][:, 2]
            h_i = df['hii']
            CD_arr = np.square(y_a - y_f) / s2_e / \
                (k - 1) * h_i / np.square(1 - h_i)
            CD = np.array(CD_arr)
            df_cd = pd.DataFrame(CD, columns=['CD'])
            filter = (df_cd['CD'] > 1)
            if (len(df_cd['CD'].loc[filter]) == 0):
                print(Emphasize("No Influential Observations by Cook's Distances"))
            else:
                print("Influential Observations by Cook's Distances =\n")
                print(df_cd['CD'].loc[filter])

    def interpret(self):
        print(color.BOLD, end='')
        print("We don\'t interpret the intercept b0 (%.4f) as the %s when every independent variable is zero. Because the sample did not cover the \"zero\" condition, we have no basis for interpreting b0." % (
            self.b[0], self.target_title['y']))
        for i, k in enumerate(self.b[1:], 1):
            if (k >= 0):
                print("b" + str(i) + " (%.4f) means that for each additional unit of %s, the %s increases by an average of %.4f." %
                      (self.b[i], self.x_name[i], self.target_title['y'], k))
            else:
                print("b" + str(i) + " (%.4f) means that for each additional unit of %s, the %s decreases by an average of %.4f." %
                      (self.b[i], self.x_name[i], self.target_title['y'], -k))
        print(color.END, end='')


class Regression_L(Regression_M):
    def __init__(self, df, target_title={}, alpha=0.05):
        self.df = df
        self.sample_size = self.df.shape[0]
        self.y = self.df[target_title['y']].dropna()
        self.x = [[1.0]]
        self.x_name = ['']
        self.x_df = pd.DataFrame([1.0] * self.sample_size, columns={'const'})
        for i in target_title:
            if (i == 'y'):
                continue
            self.x_name.append(target_title[i])
            self.x.append(self.df[target_title[i]].dropna())
            self.x_df[target_title[i]] = self.x[-1]
        self.independent_variable = len(self.x) - 1  # x數
        self.target_title = target_title
        self.result = sm.Logit(self.y, self.x_df).fit()
        self.b = []
        for i, k in enumerate(self.result.params):
            self.b.append(self.result.params[i])


class Regression_G(Regression_M):
    def __init__(self, df, target_title={}, poly=[], interaction=[], indicator={}, VT=False, alpha=0.05):
        self.df = df.copy()
        self.sample_size = self.df.shape[0]
        self.y = self.df[target_title['y']].dropna()
        self.x = [[1.0]]
        self.x_name = ['']
        self.x_df = pd.DataFrame([1.0] * self.sample_size, columns=['const'])
        self.poly = poly
        self.interaction = interaction
        self.indicator = indicator
        self.qualitative_number = 0
        for i in target_title:
            if (VT):
                if (target_title[i] in poly):
                    self.df[target_title[i]] = self.df[target_title[i]
                                                       ] - self.df[target_title[i]].mean()
            if (i == 'y'):
                continue
            self.x_name.append(target_title[i])
            self.x.append(self.df[target_title[i]].dropna())
            self.x_df[target_title[i]] = self.x[-1]
        if (indicator):
            for i, k in enumerate(indicator['qualitative']):
                self.dummy = pd.get_dummies(df[k], prefix=indicator['unit'][i])
                self.qualitative_number += len(indicator['name'][i]) - 1
                if (self.dummy.shape[1] > 2):
                    self.df = pd.concat([self.df, self.dummy], axis=1)
                    if (indicator.__contains__('designate') and indicator['designate'][i]):
                        d = [x for x in self.dummy.columns.values if x !=
                             indicator['designate'][i]]
                    else:
                        d = self.dummy.columns.values[0:-1]
                    for j in d:
                        self.x_name.append(j)
                        self.x.append(self.df[j].dropna())
                        self.x_df[j] = self.x[-1]
                else:
                    self.x_name.append(k)
                    self.x.append(self.df[k].dropna())
                    self.x_df[k] = self.x[-1]
        self.independent_variable = len(self.x) - 1  # x數
        for i in poly:
            self.x_df[i+'^2'] = self.x_df[i] ** 2
            self.x.append(self.x_df[i+'^2'])
            self.x_name.append(i+'^2')
            self.df[i+'^2'] = self.df[i] ** 2
        for i in interaction:
            self.x_df[i[0]+'_'+i[1]] = self.df[i[0]] * self.df[i[1]]
            self.x.append(self.x_df[i[0]+'_'+i[1]])
            self.x_name.append(i[0]+'_'+i[1])
            self.df[i[0]+'_'+i[1]] = self.df[i[0]] * self.df[i[1]]
        self.target_title = target_title
        self.result = sm.OLS(self.y, self.x_df).fit()
        self.residual = sso.summary_table(self.result, alpha=0.05)[1][:, 10]
        self.b = []
        for i, k in enumerate(self.result.params):
            self.b.append(self.result.params[i])
        self.r2 = round(self.result.rsquared, 4)
        self.adjusted_r2 = round(self.result.rsquared_adj, 4)
        self.standard_error = self.result.mse_resid ** 0.5
        self.term = self.independent_variable + len(poly) + len(interaction)

    def forward_selection(self):
        selected = []
        candidate = self.x_name[1:].copy()
        best_adjr2 = -1
        best_subset = []
        tmp_adjr2 = -1
        while (True):
            subset_candidate = []
            adjr2_candidate = []
            for i in candidate:
                tmp_x = selected.copy()
                tmp_x.append('Q("%s")' % i)
                modelstr = ('Q("%s")' %
                            self.target_title['y']) + " ~ " + "+".join(tmp_x)
                result = smf.ols(modelstr, data=self.df).fit()
                subset_candidate.append(tmp_x)
                adjr2_candidate.append(result.rsquared_adj)
            index = np.array(adjr2_candidate).argmax()
            tmp_adjr2 = adjr2_candidate[index]
            selected = subset_candidate[index]
            if (tmp_adjr2 <= 0):
                raise ("Encounterd negative Adj R2. Stop.")
            if (tmp_adjr2 > best_adjr2):
                best_adjr2 = tmp_adjr2
                best_subset = selected
                candidate = set(candidate) - set(selected)
                candidate = list(candidate)
            else:
                break
        delim = 'Q()\"'
        best_subset = [''.join(i for i in x if i not in delim)
                       for x in best_subset]
        print("After forward selection, the decision subset is", best_subset)
        new_target_title = {'y': self.target_title['y']}
        new_poly = []
        new_interaction = []
        count = 0
        for i in best_subset:
            if ('^2' in i):
                new_poly.append(i)
#             elif(): #qualitative
#             elif(): #interaction
            else:
                count += 1
                new_target_title['x'+str(count)] = i

    def scatter(self, sep, fontsize=6, spot=1):
        if (self.poly):
            def objective(x, a, b, c):
                return a * x + b * x**2 + c
            for i in range(1, self.independent_variable + 1):
                ax = plt.subplot(
                    (math.ceil(self.independent_variable / 3)), 3, i)
                (a, b, c), _ = curve_fit(objective, self.x[i], self.y)
                fig = plt.scatter(self.x[i], y=self.y, s=spot)
                x_line = np.arange(min(self.x[i]), max(self.x[i]), 1)
                y_line = objective(x_line, a, b, c)
                plt.plot(x_line, y_line, '--', color='red')
                ax.text(max(self.x[i]), max(self.y), 'y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c),
                        horizontalalignment='right', verticalalignment='top', fontsize=8)
                plt.title('Scatter Plot for %s and %s' % (
                    self.target_title['y'], self.x_name[i]), fontsize=fontsize * 1.5)
                plt.xlabel(self.x_name[i], fontsize=fontsize * 1.5)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.ylabel(self.target_title['y'], fontsize=fontsize * 1.5)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                                top=0.9, wspace=0.4, hspace=0.4)
            plt.show()
        elif (self.indicator):
            import random
            for k in range(len(self.indicator['qualitative'])):
                plt.figure()
                # print("Among %s:" % self.indicator['qualitative'][k])
                value = sorted(
                    self.df[self.indicator['qualitative'][k]].unique())
                # print(value)
                Color = []
                while not (len(Color) == (len(value))):
                    a = random.randint(0, 9)
                    if not (a in Color):
                        Color.append(a)
                for i in range(1, self.independent_variable + 1 - self.qualitative_number):
                    ax = plt.subplot(
                        (math.ceil(self.independent_variable / 3)), 3, i)
                    for j in range(len(value)):
                        tmp = self.df[self.df[self.indicator['qualitative'][k]] == value[j]]
                        fig = sns.regplot(x=self.x_name[i], y=self.target_title['y'], data=tmp,
                                          color="C" + str(Color[j]),
                                          ci=None, scatter_kws={'s': spot}, label=self.indicator['name'][k][j])
                    plt.xlim(self.x[i].min() * 0.95, self.x[i].max() * 1.05)
                    plt.title('Scatter Plot for %s and %s' % (
                        self.target_title['y'], self.x_name[i]), fontsize=fontsize * 1.5)
                    plt.xlabel(self.x_name[i], fontsize=fontsize * 1.5)
                    plt.xticks(fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    plt.legend(bbox_to_anchor=(1.0, 1.0),
                               loc='upper left', fontsize=fontsize)
                    # plt.tight_layout()
                    plt.ylabel(self.target_title['y'], fontsize=fontsize * 1.5)
                plt.subplots_adjust(left=0, bottom=0.1,
                                    right=0.9, top=0.9, wspace=0.5, hspace=0.4)
                plt.show()
        else:
            if (sep):
                super().scatter(True, fontsize, spot)
            else:
                import random
                for i in range(1, self.independent_variable + 1):
                    fig = sns.regplot(x=self.x[i], y=self.y, data=self.df,
                                      color=list(matplotlib.colors.cnames.values())[
                        random.randint(i, 140)],
                        ci=None, scatter_kws={'s': spot}, label=self.x_name[i])
#                     plt.xlim(self.x[i].min() * 0.95, self.x[i].max() * 1.05)
                    plt.xticks(fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                plt.xlim(self.x_df[self.x_name[1:self.independent_variable + 1]].min().min(
                ) * 0.99, self.x_df[self.x_name[1:self.independent_variable + 1]].max().max() * 1.01)
                plt.xlabel('Independent Variables')
                plt.legend()
                plt.show()
        return plt

    def line(self, plot=True, sep=True, fontsize=6, spot=1):
        if (plot):
            fig = self.scatter(sep, fontsize, spot)
        # print(color.BOLD + "Estimated model: %s = %0.4f" %
        #       (self.target_title['y'], self.b[0]), end='')
        # for i, k in enumerate(self.x_name[1:], 1):
        #     print(" + %0.4f %s" % (self.b[i], k), end='')
        # print(color.END + '\n')
        return fig

    def beta_test(self, tail, alpha=0.05):
        for i in range(1, self.term + 1):
            if (i == 1):
                print('b'+str(i)+' (%s):' % (self.x_name[i]))
            else:
                print('\nb'+str(i)+' (%s):' % (self.x_name[i]))
            self.tail = tail
            self.test = 't'
            T = self.result.tvalues[i]
            self.dof = self.result.df_resid
            tcv = self.critical_value(alpha, self.result.df_resid)
            pvalue = self.result.pvalues[i]
            print("T-stat = %0.4f, p-value = %0.4f" % (T, pvalue))
            print("T critical value %s tail =" % (tail), tcv,
                  "(degree of freedom = %d)" % self.result.df_resid)
            if (pvalue < 0.05):
                print(color.BOLD + "Since the p value %0.4f < 0.05, there is enough evidence to reject the null hypothesis：" % pvalue +
                      Emphasize("We can infer that it is linearly related to ", "can") + color.BOLD + self.target_title['y'] + color.END)
            else:
                print(color.BOLD + "Since the p value %0.4f > 0.05, there is enough evidence to reject the null hypothesis：" % pvalue +
                      Emphasize("We can\'t infer that it is linearly related to ", "can\'t") + color.BOLD + self.target_title['y'] + color.END)
            del self.tail
            del self.test
