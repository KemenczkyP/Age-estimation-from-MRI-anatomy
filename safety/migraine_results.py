# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:39:02 2019

This script compares chrnological and predicted brain age, and the brain-predicted
age difference score (brain-PAD) between two groups.

@author: Pál Vakli (RCNS HAS BIC)
"""
################################# Settings #################################### 
# _osszes / _ratanitott / empty string
dataset_id = '_ratanitott'

############################### Load & import #################################
# Import necessary modules and packages
import pickle
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# Load migraine data
with open('Migrene0426'+'.p', 'rb') as f:
    save = pickle.load(f)
    mig_truelabels = save['labels']
    mig_predlabels = save['predictions']
    
# Load control data
with open('NOTMigrene0426'+ '.p', 'rb') as f:
    save = pickle.load(f)
    ctr_truelabels = save['labels']
    ctr_predlabels = save['predictions']

######################### Distribution of true labels #########################
# Descriptives for true labels
mean_mig_truelabels = np.mean(mig_truelabels)
var_mig_truelabels = np.var(mig_truelabels)
skew_mig_truelabels = stats.skew(mig_truelabels)
kurt_mig_truelabels = stats.kurtosis(mig_truelabels)

mean_ctr_truelabels = np.mean(ctr_truelabels)
var_ctr_truelabels = np.var(ctr_truelabels)
skew_ctr_truelabels = stats.skew(ctr_truelabels)
kurt_ctr_truelabels = stats.kurtosis(ctr_truelabels)

# Subplot histograms
bins = list(range(0, 100))
f, axes = plt.subplots(1, 2, sharey=True)
sns.distplot(a=mig_truelabels, bins=bins, ax=axes[0])
sns.distplot(a=ctr_truelabels, bins=bins, ax=axes[1])

# Set subplot axis labels and titles
axes[0].set_title('Migraine group')
axes[1].set_title('Control group')
axes[0].set_xlabel('Chronological age (years)')
axes[1].set_xlabel('Chronological age (years)')
axes[0].set_ylabel('Probability density')
f.suptitle('Distribution of chronological age for '+dataset_id)

# Display additional info for subplots
axes[0].text(80, 0.1, 'Mean: '+str(round(mean_mig_truelabels, 2)), fontsize=9)
axes[0].text(80, 0.095, 'Variance: '+str(round(var_mig_truelabels, 2)), fontsize=9)
axes[0].text(80, 0.09, 'Skewness: '+str(round(skew_mig_truelabels, 2)), fontsize=9)
axes[0].text(80, 0.085, 'Kurtosis: '+str(round(kurt_mig_truelabels, 2)), fontsize=9)

axes[1].text(80, 0.1, 'Mean: '+str(round(mean_ctr_truelabels, 2)), fontsize=9)
axes[1].text(80, 0.095, 'Variance: '+str(round(var_ctr_truelabels, 2)), fontsize=9)
axes[1].text(80, 0.09, 'Skewness: '+str(round(skew_ctr_truelabels, 2)), fontsize=9)
axes[1].text(80, 0.085, 'Kurtosis: '+str(round(kurt_ctr_truelabels, 2)), fontsize=9)

################# Correlation between true and predicted labels ###############
# Calculate range of true labels
all_truelabels = np.concatenate((mig_truelabels, ctr_truelabels), axis=0)
min_truelabels = np.min(all_truelabels)
max_truelabels = np.max(all_truelabels)
range_truelabels = max_truelabels-min_truelabels

# Calculate MAE
ctr_mae = np.mean(np.abs(np.subtract(ctr_predlabels, ctr_truelabels)))
mig_mae = np.mean(np.abs(np.subtract(mig_predlabels, mig_truelabels)))

# Calculate Pearson correlation between true and predicted labels
r_mig, p_mig = stats.stats.pearsonr(mig_truelabels, mig_predlabels)
r_ctr, p_ctr = stats.stats.pearsonr(ctr_truelabels, ctr_predlabels)

# Fit 1st order polynomial to true labels
m_mig, b_mig = np.polyfit(mig_truelabels, mig_predlabels, 1)
m_ctr, b_ctr = np.polyfit(ctr_truelabels, ctr_predlabels, 1)

# Scatter subplots: true and predicted labels
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(mig_truelabels, mig_predlabels)
ax2.scatter(ctr_truelabels, ctr_predlabels)

# Plot regression lines
ax1.plot(mig_truelabels, m_mig*mig_truelabels+b_mig, 'b-')
ax2.plot(ctr_truelabels, m_ctr*ctr_truelabels+b_ctr, 'b-')

ax1.plot(np.arange(np.min(mig_truelabels),np.max(mig_truelabels)), np.arange(np.min(mig_truelabels),np.max(mig_truelabels)), 'r-')
ax2.plot(np.arange(np.min(ctr_truelabels),np.max(ctr_truelabels)),np.arange(np.min(ctr_truelabels),np.max(ctr_truelabels)), 'r-')
ax1.legend(['fitted curve', 'optimal curve', 'results'])
ax2.legend(['fitted curve', 'optimal curve', 'results'])


# Set subplot axis limits
ax1.set_xlim([min_truelabels-0.05*range_truelabels, max_truelabels+0.05*range_truelabels])
ax2.set_xlim([min_truelabels-0.05*range_truelabels, max_truelabels+0.05*range_truelabels])

# Set subplot axis labels and titles
ax1.set_title('Migraine group; r = '+str(round(r_mig, 2))+', p = '+str(round(p_mig, 4))+\
              '\nMAE = '+str(round(mig_mae, 4)))
ax2.set_title('Control group; r = '+str(round(r_ctr, 2))+', p = '+str(round(p_ctr, 4))+\
              '\nMAE = '+str(round(ctr_mae, 4)))
ax1.set_xlabel('Chronological age (years)')
ax2.set_xlabel('Chronological age (years)')
ax1.set_ylabel('Predicted age (years)')
f.suptitle('Pearson correlation between true and predicted age for '+dataset_id)

################# Brain-predicted age difference (Brain-PAD) ##################
# Calculate brain-PAD scores
mig_bpads = mig_predlabels-mig_truelabels
ctr_bpads = ctr_predlabels-ctr_truelabels

# Descriptives for brain-PAD scores
mean_mig_bpads = np.mean(mig_bpads)
var_mig_bpads = np.var(mig_bpads)
skew_mig_bpads = stats.skew(mig_bpads)
kurt_mig_bpads = stats.kurtosis(mig_bpads)

mean_ctr_bpads = np.mean(ctr_bpads)
var_ctr_bpads = np.var(ctr_bpads)
skew_ctr_bpads = stats.skew(ctr_bpads)
kurt_ctr_bpads = stats.kurtosis(ctr_bpads)

# Kolmogorov-Smirnov test for normal distribution
w_mig_bpad, wp_mig_bpad = stats.shapiro(mig_bpads)
w_ctr_bpad, wp_ctr_bpad = stats.shapiro(ctr_bpads)

# Calculate range of brain-PAD scores
all_bpads = np.concatenate((mig_bpads, ctr_bpads), axis=0)
min_bpads = np.min(all_bpads)
max_bpads = np.max(all_bpads)
range_bpads = max_bpads-min_bpads

# Subplot histograms
bins = list(range(-50, 50))
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(2, 2, 1)
sns.distplot(a=mig_bpads, bins=bins, ax=ax1)
ax2 = fig.add_subplot(2, 2, 2)
sns.distplot(a=ctr_bpads, bins=bins, ax=ax2)
ax3 = fig.add_subplot(2, 2, 3)
stats.probplot(mig_bpads, dist='norm', plot=ax3)
ax4 = fig.add_subplot(2, 2, 4)
stats.probplot(ctr_bpads, dist='norm', plot=ax4)

# Set subplot axis labels and titles
ax1.set_title('Migraine group --Shapiro-Wilk w = '+str(round(w_mig_bpad, 4))+', p = '+str(round(wp_mig_bpad, 4)))
ax2.set_title('Control group --Shapiro-Wilk w = '+str(round(w_ctr_bpad, 4))+', p = '+str(round(wp_ctr_bpad, 4)))
ax1.set_xlabel('Brain-PAD (years)')
ax2.set_xlabel('Brain-PAD (years)')
ax1.set_ylabel('Probability density')
ax3.set_title('Q-Q plot')
ax4.set_title('Q-Q plot')
fig.suptitle('Distribution of brain-PAD scores for '+dataset_id)

# Display additional info for subplots
ax1.text(30, 0.1, 'Mean: '+str(round(mean_mig_bpads, 2)), fontsize=9)
ax1.text(30, 0.095, 'Variance: '+str(round(var_mig_bpads, 2)), fontsize=9)
ax1.text(30, 0.09, 'Skewness: '+str(round(skew_mig_bpads, 2)), fontsize=9)
ax1.text(30, 0.085, 'Kurtosis: '+str(round(kurt_mig_bpads, 2)), fontsize=9)

ax2.text(30, 0.1, 'Mean: '+str(round(mean_ctr_bpads, 2)), fontsize=9)
ax2.text(30, 0.095, 'Variance: '+str(round(var_ctr_bpads, 2)), fontsize=9)
ax2.text(30, 0.09, 'Skewness: '+str(round(skew_ctr_bpads, 2)), fontsize=9)
ax2.text(30, 0.085, 'Kurtosis: '+str(round(kurt_ctr_bpads, 2)), fontsize=9)

### Pearson correlation between chronological age and brain-PAD
r_chronbpad_mig, p_chronbpad_mig = stats.stats.pearsonr(mig_truelabels, mig_bpads)
r_chronbpad_ctr, p_chronbpad_ctr = stats.stats.pearsonr(ctr_truelabels, ctr_bpads)

# Fit 1st order polynomial to true labels
m_chronbpad_mig, b_chronbpad_mig = np.polyfit(mig_truelabels, mig_bpads, 1)
m_chronbpad_ctr, b_chronbpad_ctr = np.polyfit(ctr_truelabels, ctr_bpads, 1)

# Scatter subplots: true labels and brain-PAD scores
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(mig_truelabels, mig_bpads)
ax2.scatter(ctr_truelabels, ctr_bpads)

# Plot regression lines
ax1.plot(mig_truelabels, m_chronbpad_mig*mig_truelabels+b_chronbpad_mig, 'b-')
ax2.plot(ctr_truelabels, m_chronbpad_ctr*ctr_truelabels+b_chronbpad_ctr, 'b-')

ax1.plot(np.arange(np.min(mig_truelabels),np.max(mig_truelabels)),
         np.zeros_like(np.arange(np.min(mig_truelabels),np.max(mig_truelabels))), 'r-')
ax2.plot(np.arange(np.min(ctr_truelabels),np.max(ctr_truelabels)),
         np.zeros_like(np.arange(np.min(ctr_truelabels),np.max(ctr_truelabels))), 'r-')
ax1.legend(['fitted curve', 'optimal curve', 'results'])
ax2.legend(['fitted curve', 'optimal curve', 'results'])


# Set subplot axis limits
ax1.set_xlim([min_truelabels-0.05*range_truelabels, max_truelabels+0.05*range_truelabels])
ax2.set_xlim([min_truelabels-0.05*range_truelabels, max_truelabels+0.05*range_truelabels])



# Set subplot axis labels and titles
ax1.set_title('Migraine group; r = '+str(round(r_chronbpad_mig, 2))+', p = '+str(round(p_chronbpad_mig, 4)))
ax2.set_title('Control group; r = '+str(round(r_chronbpad_ctr, 2))+', p = '+str(round(p_chronbpad_ctr, 4)))
ax1.set_xlabel('Chronological age (years)')
ax2.set_xlabel('Chronological age (years)')
ax1.set_ylabel('Brain-PAD score')
f.suptitle('Pearson correlation between chronological age and brain-PAD scores for '+dataset_id)

##################### Comparing brain-PAD between groups ######################
# Levene's Test for Equality of Variances
levene_w, levene_p = stats.levene(ctr_bpads, mig_bpads)

# Independent sample t-test
t, p = stats.ttest_ind(ctr_bpads, mig_bpads)

# Print descriptive statistics and the results of statistical inference
print('----------------------------------------------------------------------')
print('Descriptive statistics for brain-PAD: ')
print('     Control group: ')
print('         N = %.d' % ctr_truelabels.shape[0])
print('         Mean: %.4f' % np.mean(ctr_bpads))
print('         Variance: %.4f' % np.var(ctr_bpads))
print('     Migraine group: ')
print('         N = %.d' % mig_truelabels.shape[0])
print('         Mean: %.4f' % np.mean(mig_bpads))
print('         Variance: %.4f' % np.var(mig_bpads))
print('Levene\'s Test for Equality of Variances: ')
print('     W = %.4f' % levene_w)
print('     p = %.4f' % levene_p)
print('Independent sample t-test: ')
print('     t = %.4f' % t)
print('     p = %.4f' % p)
print('----------------------------------------------------------------------')

# Violinplot
all_bpads = np.concatenate((mig_bpads, ctr_bpads), axis=0).reshape((-1, 1))
group_inds = np.concatenate((np.zeros((mig_bpads.shape[0], 1)), np.zeros((ctr_bpads.shape[0], 1))+1), axis=0)
data = np.concatenate((all_bpads, group_inds), axis=1)
df = pd.DataFrame(data, columns=['bpads', 'groups'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.violinplot(x="groups", y="bpads", data=df, ax=ax)
ax.set_xticklabels(['migraine', 'control'])
ax.set_ylabel('Brain-PAD scores')