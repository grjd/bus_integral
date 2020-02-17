#######################################################
#Python program name	: 
#Description	: bus_regresslong.py
#Args           : Code for buschke_integral paper                                       
#Author       	: Jaime Gomez-Ramirez                                               
#Email         	: jd.gomezramirez@gmail.com 
#REMEMBER to Activate Keras source ~/github/code/tensorflow/bin/activate
#pyenv install 3.7.0
#pyenv local 3.7.0
#python3 -V
# To use ipython3 debes unset esta var pq contien old version
#PYTHONPATH=/usr/local/lib/python2.7/site-packages
#unset PYTHONPATH
# $ipython3
# To use ipython2 /usr/local/bin/ipython2
#/Library/Frameworks/Python.framework/Versions/3.7/bin/ipython3
#pip install rfpimp. (only for python 3)
#Last Modification: 21 Jan 2020
#######################################################
# -*- coding: utf-8 -*-

import sys, os, pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
sys.path.append('/Users/jaime/github/papers/atrophy_long/code')
from atrophyLong_paper import convert_stringtofloat, scatterplot_2variables_in_df
#sys.path.append('/Users/jaime/github/papers/JADr_VallecasIndex/code/')
#from JADr_paper import ROC_Curve


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
	path = os.path.join(figures_dir, fig_id + "." + fig_extension)
	print("Saving figure", fig_id)
	if tight_layout:
	    plt.tight_layout()
	plt.savefig(path, format=fig_extension, dpi=resolution)

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    from scipy.spatial.distance import pdist, squareform
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def remove_outliers(df, cols):
	"""remove_outliers dataframe, cols
	"""
	df_orig = df.copy()
	low = .01
	high = .99
	dfcols = df[cols]
	quant_df = df[cols].quantile([low, high])
	
	print('Outliers: low= %.3f high= %.3f \n %s' %(low, high, quant_df))
	df_nooutliers = dfcols[(dfcols > quant_df.loc[low, cols]) & (dfcols < quant_df.loc[high, cols])]
	df_outs = dfcols[(dfcols <= quant_df.loc[low, cols]) | (dfcols >= quant_df.loc[high, cols])]
	df[cols] = df_nooutliers.to_numpy()
	# List of outliers
	reportfile = os.path.join(figures_dir, 'outliersHippo.txt')
	file_h= open(reportfile,"w+")
	print('Hippocampal Outliers Low %s: High  %s' %(low, high))
	file_h.write('Outliers: low= %.3f high= %.3f \n %s \n' %(low, high, quant_df))
	for year in cols:
		outliers_y = df_outs.index[df_outs[year].notna() == True].tolist()
		file_h.write('\tOutliers Years :' + year + str(outliers_y) + '\n')
	
	return df

def clean_df(df):
	"""
	"""
	df.describe()
	# Remove id/
	del df['id']
	subs = 'fcsrtrl'
	res = [i for i in df.columns if subs in i]
	print(res)
	return df 

def regression_mv(X,y):
	"""
	"""
	#X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)
	# Model initialization

	regression_model = LinearRegression()
	scores = []
	kfold = KFold(n_splits=3, shuffle=True, random_state=42)
	for i, (train, test) in enumerate(kfold.split(X, y)):

		regression_model.fit(X.iloc[train,:], y.iloc[train,:])
		score = regression_model.score(X.iloc[test,:], y.iloc[test,:])
		scores.append(score); print(scores)
	print(scores)
	#regression_model.coef_
	#regression_model.intercept_
	#regression_model.predict(np.array([[15]]))
	# The coefficients
	print('Coefficients: \n', regression_model.coef_)
	print('Intercept : \n', regression_model.intercept_)

	# Make predictions using the testing set
	y_pred = regression_model.predict(X.iloc[test,:])
	# The coefficients
	print('Coefficients: \n', regression_model.coef_)
	# The mean squared error
	print('Mean squared error: %.2f' % mean_squared_error(y.iloc[test,:], y_pred))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.2f' % r2_score( y.iloc[test,:], y_pred))
	# Plot outputs
	plt.scatter(X.iloc[test,:], y.iloc[test,:],  color='black')
	# plt.plot(X.iloc[test,:], y.iloc[test,:], color='blue', linewidth=3)
	# plt.xticks(())
	# plt.yticks(())
	# fig_file = os.path.join(figures_dir, 'scatter_B.png')
	# plt.savefig(fig_file)
	return regression_model


def buschke_aggregate(y):
	"""buschke_aggregate: computes the Buschke aggregate 
	Args: y array or list of len 3
	Output:
	"""
	from scipy.integrate import trapz, simps
	from scipy.interpolate import interp1d
	from matplotlib.patches import Polygon
	# The rank of the coefficient matrix in the least-squares fit is deficient. The warning is only raised if full = False.
	# Turn off warning
	#polyfit issues a RankWarning when the least-squares fit is badly conditioned. This implies that the best fit is not well-defined due to numerical error
	import warnings
	warnings.simplefilter('ignore', np.RankWarning)
	from scipy.interpolate import UnivariateSpline
	
	npx, degree = 3, 2
	x = np.array([0,1,2])
	if y is None: y = np.array([0,1,0])
	pointspersector = 100
	interp_points = (npx-1)*pointspersector
	xs = np.linspace(0, npx-1, interp_points)
	if type(y) is list:
		y = np.asarray(y)
	z_fit = np.polyfit(x[:], y[:], deg=degree)
	p_fit = np.poly1d(z_fit)

	area_tot =  simps(y[0:2], x[0:2], even='avg') + simps(y[1:], x[1:], even='avg')
	area_f = simps(y[0:2] -y[0], x[0:2], even='avg')
	area_s = simps(y[1:] -y[1], x[1:], even='avg')
	b_agg = area_tot + area_f + area_s
	print(y, area_tot, area_f,area_s, 'b_agg=', b_agg, '\n')
	ymin, ymax = 0 - 0.1, max(y)+1
	plot_fitted_pol  = False
	if plot_fitted_pol is True:
		# fit polynomial of degree that pass for (x[1:-1], b_list) points
		# z highest power first
		fig, axes = plt.subplots(1, 2)
		axes[0].plot(x, y, '.', xs, p_fit(xs), '-')
		axes[0].set_xticks([0,1,2])
		axes[0].set_title('polyfit')
		#axes[0].set_xlim([xmin,xmax])
		axes[0].set_ylim([ymin,ymax])
		axes[0].grid(True)
		# Degree of the smoothing spline. Must be <= 5. Default is k=3, a cubic spline
		spl = UnivariateSpline(x, y, k=2)
		axes[1].plot(xs, spl(xs), 'b', lw=3)
		axes[1].set_xticks([0,1,2])
		#axes[1].set_xlim([xmin,xmax])
		axes[1].set_ylim([ymin,ymax])
		axes[1].set_title('Spline')
		axes[1].grid(True)
	
	b_values = [b_agg, area_tot, area_f, area_s, z_fit, p_fit]
	return b_values

def compute_buschke_integral(dataframe, yf=6, features_dict=None):
	""" compute_buchske_integral_df compute new Buschke 
	Args: dataframe with the columns fcsrtrl1_visita[1-7]
	Output:return the dataframe including the columns bus_visita[1-7]"""

	import scipy.stats.mstats as mstats
	print('Compute the Buschke aggregate \n')
	# Busche integral (aggregate is calculated with 3 values, total integral and partial integral 21 and 32)
	S = [0] * dataframe.shape[0]
	S21 = [0] * dataframe.shape[0]
	S32 = [0] * dataframe.shape[0]
	# arithmetic, gemoetric mean and sum of Bischke scores
	mean_a, mean_g, suma = S[:], S[:], S[:]
	#longit_pattern= re.compile("^fcsrtrl[1-3]_+visita+[1-7]")
	#longit_status_columns = [x for x in dataframe.columns if (longit_pattern.match(x))]
	for i in [1, yf]:
		coda='visita'+ format(i)
		#bus_scores = ['fcsrtrl1_visita1', 'fcsrtrl2_visita1', 'fcsrtrl3_visita1']
		bus_scores = ['fcsrtrl1_'+coda, 'fcsrtrl2_'+coda,'fcsrtrl3_'+coda]
		df_year = dataframe[bus_scores]
		df_year = df_year.values
		#bus_scores = ['fcsrtrl1_visita2', 'fcsrtrl2_visita2', 'fcsrtrl3_visita2']
		for ix, y in enumerate(df_year):	
			#print(row[bus_scores[0]], row[bus_scores[1]],row[bus_scores[2]])
			#bes == [b_agg(suma de scors), area_tot (are definidia por la curva), area_f (area1), area_s (area2), z_fit, p_fit (2 ans 1 pol fit)]
			bes = buschke_aggregate(y)
			S[ix]=bes[0] #sum of scores
			S21[ix]=bes[2] #definite integral first part
			# bes[1] is the area of the curve
			S32[ix]=bes[3] #definite integral second part
			mean_a[ix] = np.mean(y)
			mean_g[ix] = mstats.gmean(y)
			suma[ix] = np.sum(y) #==bes[0]
			print('Total Aggregate S=', bes[0])
			print('Total sum Sum =', bes[1], ' partial Sum 10=',bes[2], ' Partial Sum 21=',bes[3])
			print('arithmetic mean:', mean_a[ix], ' Geometric mean:', mean_g[ix], ' Sum:',suma[ix])
			print('Poly1d exponents decreasing' ,bes[-1])
			print('Poly2 exponents decreasing',bes[-2])
			print('\n')

		coda_col= 'bus_int_'+ coda
		dataframe[coda_col] = S
		dataframe['bus_parint1_' + coda] = S21
		dataframe['bus_parint2_' + coda] = S32
		dataframe['bus_sum_'+ coda] = suma
		dataframe['bus_meana_'+coda] = mean_a
	# int + parint1 + parint2 (Integral 1-3 + pIntegral 12 + pIntegral 23) 
	# versus sum
	print('Created Buschke Integral variables bus_parint1_%s,bus_parint2_,bus_sum_, bus_meana_' %coda)
	return dataframe


def normalize_Bus(df):
	"""
	"""
	#from sklearn import preprocessing
	bus_cols = ['bus_parint1_visita1', 'bus_parint2_visita1', 'bus_sum_visita1','bus_parint1_visita6', 'bus_parint2_visita6', 'bus_sum_visita6']
	for label in bus_cols:
		#df['bus_parint1_visita1'] = (df['bus_parint1_visita1'] - df['bus_parint1_visita1'].mean())/df['bus_parint1_visita1'].std(ddof=0)
		print('Normalizing %s ' %label) 
		df[label] = (df[label] - df[label].mean())/df[label].std(ddof=0)
	return df

def plot_Bus_distros(df, cols2plotlist=None):
	"""
	"""
	cols2plot = cols2plotlist[0] #['bus_parint1_visita1','bus_parint1_visita6']
	yearfinal = cols2plot[-1][-1]
	fig, ax = plt.subplots(1,3)
	df[cols2plot].plot.kde(ax=ax[0], legend=False, title='ParInt1 yy: 1 vs. ' + yearfinal)
	df[cols2plot].plot.hist(density=True, ax=ax[0],legend=False)
	#ax[0].set_ylabel('Def Integral 1st')
	ax[0].grid(axis='y')
	ax[0].set_facecolor('#d8dcd6')
	
	cols2plot = cols2plotlist[1] #['bus_parint2_visita1','bus_parint2_visita6']
	df[cols2plot].plot.kde(ax=ax[1], legend=False, title='ParInt2 yy: 1 vs. '+ yearfinal)
	df[cols2plot].plot.hist(density=True, ax=ax[1],legend=False)
	#ax[1].set_ylabel('Def Integral 2nd')
	ax[1].grid(axis='y')
	ax[1].set_facecolor('#d8dcd6')

	cols2plot = cols2plotlist[2] # cols2plot = ['bus_sum_visita1','bus_sum_visita6']
	df[cols2plot].plot.kde(ax=ax[2], legend=False, title='B.sum yy: 1 vs. '+ yearfinal)
	df[cols2plot].plot.hist(density=True, ax=ax[2],legend=True)
	#ax[2].set_ylabel('Sum B.score')
	ax[2].grid(axis='y')
	ax[2].set_facecolor('#d8dcd6')

	fig_file = os.path.join(figures_dir, 'Bus_hists.png')
	# plt.savefig(fig_file)
	fig.savefig(fig_file)

def plot_brain_distros(df, cols2plotlist=[['hippoL_y6y1'],['hippoR_y6y1']]):
	"""
	"""
	cols2plot = cols2plotlist[0]
	fig, ax = plt.subplots(1,2)
	df[cols2plot].plot.hist(density=True, ax=ax[0],legend=False, alpha=0.5, title='Vol Left-Hippo (6 -1) ')
	cols2plot = cols2plotlist[1]
	df[cols2plot].plot.hist(density=True, ax=ax[1],legend=False, alpha=0.5, title='Vol Right-Hippo (6 -1) ')
	fig_file = os.path.join(figures_dir, 'Hipp_hists_noLong.png')
	fig.savefig(fig_file)

def distance_correlation(x,y):
	"""
	"""
	#import scipy.spatial.distance as dis 
	#Computes the correlation distance between two 1-D arrays.
	# Székely dcor Brownian distance https://pypi.org/project/dcor/
	import dcor
	x = x.values; y = y.values
	DX = dcor.distance_correlation_sqr(x, y)
	print('DX = %.3f \n' %DX)
	return DX
	#DX = dcor.distances.pairwise_distances(x)

def OLS_regression(df):
	"""
	"""
	import statsmodels.api as sm
	#bus_parint1_visita%i ,bus_parint2_visita%i,bus_sum_visita%i
	x_cols = ['bus_parint1_visita1', 'bus_parint2_visita1', 'bus_sum_visita1','edad_visita1',\
	'bus_parint1_visita6', 'bus_parint2_visita6', 'bus_sum_visita6']
	x_cols_visita1 = x_cols[0:3]; x_cols_visita6 = x_cols[4:];
	y_col_visita1 = ['L_Hipp_visita1']; y_col_visita6 = ['L_Hipp_visita6']; 
	
	

	# get 2 new cols from 6 cols (sum, par,par -> integral)
	# Add a constant 
	X = sm.add_constant(df[x_cols_visita1])
	# Construct the model
	model_HL = sm.OLS(df[y_col_visita1], X).fit()
	model_HR = sm.OLS(df['R_Hipp_visita1'], X).fit()
	model_H2 = sm.OLS(df['L_Hipp_visita1'] + df['R_Hipp_visita1'], X).fit()

	print(model_HL.summary());print(model_HR.summary());print(model_H2.summary())

	# model atrophy versus difference in B. score
	
	df['bus1_diff'] = df['bus_parint1_visita6'].sub(df['bus_parint1_visita1'], axis = 0)
	df['bus2_diff'] = df['bus_parint2_visita6'].sub(df['bus_parint2_visita1'], axis = 0)
	df['bussum_diff'] = df['bus_sum_visita6'].sub(df['bus_sum_visita1'], axis = 0)
	#X_atrophy = [df['bus_parint1_visita6'] - df['bus_parint1_visita1'], df['bus_parint2_visita6'] - df['bus_parint2_visita1'], \
	#df['bus_sum_visita6'] - df['bus_sum_visita1']]

	df['hipp_diffL'] = df['L_Hipp_visita6'].sub(df['L_Hipp_visita1'], axis = 0)
	df['hipp_diffR'] = df['R_Hipp_visita6'].sub(df['R_Hipp_visita1'], axis = 0)
	df['hipp_diff2'] = df['hipp_diffL'].add(df['hipp_diffR'], axis = 0)
	#pdb.set_trace()
	model_atr_HL = sm.OLS(df['hipp_diffL'], df[['bus1_diff','bus2_diff','bussum_diff']]).fit()
	model_atr_HR = sm.OLS(df['hipp_diffR'], df[['bus1_diff','bus2_diff','bussum_diff']]).fit()
	model_atr_H2 = sm.OLS(df['hipp_diff2'], df[['bus1_diff','bus2_diff','bussum_diff']]).fit()
	print(model_atr_HL.summary());print(model_atr_HR.summary());print(model_atr_H2.summary())

	df = normalize_Bus(df)

def atrophy_freesurfer(df):
	""" return dataframe adding y6 - y1 all frreesurfer columsn
	"""
	colsofinterest = ['fr_BrainSegVol_y6', 'fr_BrainSegVol_y1','fr_BrainSegVol_to_eTIV_y6','fr_BrainSegVol_to_eTIV_y1','fr_Brain_Stem_y6','fr_Brain_Stem_y1','fr_CC_Central_y6','fr_CC_Central_y1','fr_CC_Anterior_y6','fr_CC_Anterior_y1','fr_CC_Posterior_y6','fr_CC_Posterior_y1','fr_CSF_y6,fr_CSF_y1','fr_CerebralWhiteMatterVol_y6','fr_CerebralWhiteMatterVol_y1','fr_CortexVol_y6','fr_CortexVol_y1,','fr_EstimatedTotalIntraCranialVol_y6','fr_EstimatedTotalIntraCranialVol_y1']
	df['BrainSegVol_diff'] = df['fr_BrainSegVol_y6'].sub(df['fr_BrainSegVol_y1'], axis = 0)
	df['BrainSegVol_to_eTIV_diff'] = df['fr_BrainSegVol_to_eTIV_y6'].sub(df['fr_BrainSegVol_to_eTIV_y1'], axis = 0)
	df['lhCortexVol_diff'] = df['fr_lhCortexVol_y6'].sub(df['fr_lhCortexVol_y1'], axis = 0)
	df['rhCortexVol_diff'] = df['fr_rhCortexVol_y6'].sub(df['fr_rhCortexVol_y1'], axis = 0)
	

	df['Brain_Stem_diff'] = df['fr_Brain_Stem_y6'].sub(df['fr_Brain_Stem_y1'], axis = 0)
	# corpus callosum
	df['CC_Central_diff'] = df['fr_CC_Central_y6'].sub(df['fr_CC_Central_y1'], axis = 0)
	df['CC_Anterior_diff'] = df['fr_CC_Anterior_y6'].sub(df['fr_CC_Anterior_y1'], axis = 0)
	df['CC_Posterior_diff'] = df['fr_CC_Posterior_y6'].sub(df['fr_CC_Posterior_y1'], axis = 0)
	# Tissue
	df['CSF_diff'] = df['fr_CSF_y6'].sub(df['fr_CSF_y1'], axis = 0)
	df['CerebralWhiteMatterVol_diff'] = df['fr_CerebralWhiteMatterVol_y6'].sub(df['fr_CerebralWhiteMatterVol_y1'], axis = 0)
	df['TotalGrayVol_diff'] = df['fr_TotalGrayVol_y6'].sub(df['fr_TotalGrayVol_y1'], axis = 0) 
	df['SubCortGrayVol_diff'] = df['fr_SubCortGrayVol_y6'].sub(df['fr_SubCortGrayVol_y1'], axis = 0) 
	# ICV
	df['CortexVol_y1_diff'] = df['fr_CortexVol_y6'].sub(df['fr_CortexVol_y1'], axis = 0)
	df['EstimatedTotalIntraCranialVol_diff'] = df['fr_EstimatedTotalIntraCranialVol_y6'].sub(df['fr_EstimatedTotalIntraCranialVol_y1'], axis = 0)

	# Structures 
	df['Accumbens_Left_diff'] = df['fr_Left_Accumbens_area_y6'].sub(df['fr_Left_Accumbens_area_y1'], axis = 0)
	df['Accumbens_Right_diff'] = df['fr_Right_Accumbens_area_y6'].sub(df['fr_Right_Accumbens_area_y1'], axis = 0)
	df['Amygdala_Left_diff'] = df['fr_Left_Amygdala_y6'].sub(df['fr_Left_Amygdala_y1'], axis = 0)
	df['Amygdala_Right_diff'] = df['fr_Right_Amygdala_y6'].sub(df['fr_Right_Amygdala_y1'], axis = 0)

	df['Caudate_Left_diff'] = df['fr_Left_Caudate_y6'].sub(df['fr_Left_Caudate_y1'], axis = 0)
	df['Caudate_Right_diff'] = df['fr_Right_Caudate_y6'].sub(df['fr_Right_Caudate_y1'], axis = 0)
	df['Inf_Lat_Vent_Left_diff'] = df['fr_Left_Accumbens_area_y6'].sub(df['fr_Left_Accumbens_area_y1'], axis = 0)
	df['Inf_Lat_Vent_Right_diff'] = df['fr_Right_Accumbens_area_y6'].sub(df['fr_Right_Accumbens_area_y1'], axis = 0)
	df['Pallidum_Left_diff'] = df['fr_Left_Pallidum_y6'].sub(df['fr_Left_Pallidum_y1'], axis = 0)
	df['Pallidum_Right_diff'] = df['fr_Right_Pallidum_y6'].sub(df['fr_Right_Pallidum_y1'], axis = 0)
	df['Putamen_Left_diff'] = df['fr_Left_Putamen_y6'].sub(df['fr_Left_Putamen_y1'], axis = 0)
	df['Putamen_Right_diff'] = df['fr_Right_Putamen_y6'].sub(df['fr_Right_Putamen_y1'], axis = 0)
	df['Thalamus_Proper_Left_diff'] = df['fr_Left_Thalamus_Proper_y6'].sub(df['fr_Left_Thalamus_Proper_y1'], axis = 0)
	df['Thalamus_Proper_Right_diff'] = df['fr_Right_Thalamus_Proper_y6'].sub(df['fr_Right_Thalamus_Proper_y1'], axis = 0)
	
	# Destriaux
	df['fr_L_thick_G_temporal_middle_diff'] = df['fr_L_thick_G_temporal_middle_y6'].sub(df['fr_L_thick_G_temporal_middle_y1'], axis = 0)
	df['fr_R_thick_G_temporal_middle_diff'] = df['fr_R_thick_G_temporal_middle_y6'].sub(df['fr_R_thick_G_temporal_middle_y1'], axis = 0)
	df['fr_L_thick_G_precuneus_diff'] = df['fr_L_thick_G_precuneus_y6'].sub(df['fr_L_thick_G_precuneus_y1'], axis = 0)
	df['fr_R_thick_G_precuneus_diff'] = df['fr_R_thick_G_precuneus_y6'].sub(df['fr_R_thick_G_precuneus_y1'], axis = 0)
	df['fr_L_thick_Pole_temporal_diff'] = df['fr_L_thick_Pole_temporal_y6'].sub(df['fr_L_thick_Pole_temporal_y1'], axis = 0)
	df['fr_R_thick_G_precuneus_diff'] = df['fr_R_thick_G_precuneus_y6'].sub(df['fr_R_thick_G_precuneus_y1'], axis = 0)
	return df


def MLP_regressor(X, y):
	"""https://karpathy.github.io/2019/04/25/recipe/
	Adam with a learning rate of 3e-4.
	"""

	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	from tensorflow import keras
	from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error,r2_score
	
	from sklearn.neural_network import MLPRegressor
	from sklearn import metrics
 	
	scaler = StandardScaler()
	scaler = MinMaxScaler(feature_range=(0.1,4))
	
	X_train_full, X_test, y_train_full, y_test = train_test_split(X, y)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
	X_train = scaler.fit_transform(X_train) #X_train_scaled
	X_valid = scaler.transform(X_valid) #X_valid_scaled
	X_test = scaler.transform(X_test) #X_test_scaled
	hidden_layer_sizes = (100, 80, 60, 40,20)
	#activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
	model = MLPRegressor(activation='identity',hidden_layer_sizes=hidden_layer_sizes)
	model.fit(X_train, y_train)
	print(); print(model)

	y_pred = model.predict(X_test)

	# summarize the fit of the model
	expl_var = explained_variance_score(y_test, y_pred)
	max_err = max_error(y_test, y_pred)
	mean_abs_err = mean_absolute_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	print('Model Explained Variance (best is 1): %.3f' %(expl_var))
	print('Max Error (worst case true-pred) %.3f \n' %(max_err))
	print('Mean Abs Error (worst case true-pred) %.3f \n' %(max_err))
	print('R2 (Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse))%.3f \n' %(r2))

	return model

	epochs = 160; batch_size = 32
	model = keras.models.Sequential([keras.layers.Dense(20, activation="relu", input_shape=X_train.shape[1:]),\
		keras.layers.Dense(1, activation="relu")])
	modMLPRegressorel.summary()
	weights, biases = model.layers[0].get_weights()
	sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss= "mae",optimizer='nadam')
	history = model.fit(X_train, y_train.values.reshape(-1,1), validation_data=(X_valid, y_valid.values.reshape(-1,1)))
	# Model evaluation and predictions
	
	model.evaluate(X_test, y_test.values.reshape(-1,1))
	y_pred = model.predict(X_test)
	pdb.set_trace()
	y_pred.round(3)
	

	return model

def MLP_classifier(X, y):
	"""MLP_classifier: multilayer Perceptron for Classification task
	"""
	from tensorflow import keras
	from sklearn.preprocessing import StandardScaler
	from sklearn.metrics import accuracy_score, f1_score, \
	roc_auc_score, precision_recall_fscore_support,\
	classification_report, precision_score, average_precision_score
	from sklearn.utils import class_weight

	X_train_full, X_test, y_train_full, y_test = train_test_split(X, y)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train) #X_train_scaled
	X_valid = scaler.transform(X_valid) #X_valid_scaled
	X_test = scaler.transform(X_test) #X_test_scaled
	# Unbalanced Set
	cls_weights = class_weight.compute_class_weight('balanced', np.unique(y_train._values), y_train._values)
	cls_weight_dict = {0: cls_weights[0], 1: cls_weights[1]}
	val_sample_weights = class_weight.compute_sample_weight(cls_weight_dict, y_valid._values)

	epochs = 16000; batch_size = 32
	model = keras.models.Sequential([keras.layers.Dense(16, activation="selu", kernel_regularizer=keras.regularizers.l1(0.001), kernel_initializer="he_normal",input_shape=X_train.shape[1:]),\
		keras.layers.Dense(1, activation="sigmoid")])
	model.summary()
	weights, biases = model.layers[0].get_weights()
	sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss="binary_crossentropy",optimizer=sgd, weighted_metrics=['accuracy'])
	history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=cls_weight_dict, validation_data=(X_valid, y_valid))
	# Model evaluation and predictions
	model.evaluate(X_test, y_test)
	print('Model Evaluation:: Loss test/eval %.3f/%.3f' %(model.evaluate(X_test, y_test)[0], model.evaluate(X_valid, y_valid)[0]))
	print('Model Evaluation:: Acc test/eval %.3f/%.3f \n' %(model.evaluate(X_test, y_test)[1], model.evaluate(X_valid, y_valid)[1]))
	
	y_proba = model.predict(X_test)
	y_proba.round(2)
	y_pred_class = model.predict_classes(X_test)
	
	errors_pc = y_pred_class.flatten() - y_test.values
	print('Accuracy on Test Set is: %.3f, %d right %d wrong \n' %(sum(errors_pc==0)/len(errors_pc),sum(errors_pc==0) ,len(errors_pc) - sum(errors_pc==0)))
	print('False Positive (predicted 1 true 0): %d False Negative (pred 0 true 1): %d \n' %(sum(errors_pc==1),sum(errors_pc==-1)))
	# 'binary' default  report results for the 1 class 'weighted' average weighted by support 
	# 'macro':Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
	# 'micro' Calculate metrics globally by counting the total true positives, false negatives and false positives.
	average = ['micro', 'macro', 'binary', 'weighted']
	precis_test = precision_score(y_test.values, y_pred_class.flatten(),average=average[0])
	print('Precision -%s- score = %.4f' %(str(average[0]), precis_test))
	average_precision = average_precision_score(y_test, y_pred_class.flatten())
	print('Average precision-recall score: {0:0.2f}'.format(average_precision))
	
	# Plot learning
	pd.DataFrame(history.history).plot(figsize=(8, 5))
	plt.grid(True)
	plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
	plt.savefig(os.path.join(figures_dir, 'MLP_learningcurves.png'))

	# Predict model
	target_names = ['class 0', 'class 1']
	y_pred = model.predict(X_test, batch_size=32, verbose=1)
	# Classify predictions based on threshold at 0.5
	y_pred_binary = (y_pred > 0.5) * 1
	sklearn_accuracy = accuracy_score(y_test, y_pred_binary)
	sklearn_class_report = classification_report(y_test, y_pred_binary,target_names=target_names)
	test_sample_weights = class_weight.compute_sample_weight(cls_weight_dict, y_test._values)
	sklearn_weighted_accuracy = accuracy_score(y_test, y_pred_binary, sample_weight=test_sample_weights)
	
	print('sklearn_accuracy=%.3f' %sklearn_accuracy)
	print('sklearn_weighted_accuracy=%.3f' %sklearn_weighted_accuracy)
	print(sklearn_class_report)
	return model 

def plot_learning_curves(svr,svr_p, X,y):
	"""https://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-plot-kernel-ridge-regression-py
	"""

	from sklearn.model_selection import learning_curve

	#train_sizes, train_scores_svr, test_scores_svr = learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),scoring="neg_mean_squared_error", cv=10)
	fig= plt.figure(figsize=(9,9))
	#learning_curve Determines cross-validated training and test scores for different training set sizes.
	train_sizes, train_scores_svr, test_scores_svr = learning_curve(svr, X, y, train_sizes=np.linspace(0.1, 1, 30),scoring="neg_mean_squared_error", cv=10)
	#train_sizes_abs, train_scores_kr, test_scores_kr = learning_curve(svr_p, X, y, train_sizes=np.linspace(0.1, 1, 30),scoring="neg_mean_squared_error", cv=10)
	
	plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",label="SVR_RBF")
	#plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g",label="SVR_poly")
	
	plt.xlabel("Train size")
	plt.ylabel("Mean Squared Error")
	plt.title('Learning curves')
	plt.legend(loc="best")
	fig_file = os.path.join(figures_dir, 'svm_learning22.png')
	plt.savefig(fig_file)

def svm_classifier(df, xlabel, ylabel):
	"""
	"""
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.svm import LinearSVC
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.svm import SVC

	X = df[xlabel]; y = df[ylabel]
	svm_clf = Pipeline([("scaler", StandardScaler()), \
		("linear_svc", LinearSVC(C=1, loss="hinge"))])
	svm_clf.fit(X,y)
	# Non linear:: polynomial transformation
	polynomial_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree=3)),\
		("scaler", StandardScaler()), ("svm_clf", LinearSVC(C=10, loss="hinge"))])
	polynomial_svm_clf.fit(X,y)
	# Non linear:: Gaussian kernel
	rbf_svm_clf = Pipeline([("scaler", StandardScaler()), ("svm_clf", SVC(kernel="rbf", gamma = 5, C=10))])
	rbf_svm_clf.fit(X,y)
	return [svm_clf,polynomial_svm_clf,rbf_svm_clf]

def svm_regression(df, xlabel, ylabel):
	""" fit as many instances as possible on the street while limiting instances off the street
	"""
	from sklearn.svm import LinearSVR
	from sklearn.svm import SVR

	X = df[xlabel]; y = df[ylabel]
	# SVR is e insensitive (adding more training points within the margin does not affect the model's predictions)
	svm_reg = LinearSVR(epsilon=1.5)
	svm_reg.fit(X,y)
	# For non linear regression use kernelized SVM model
	# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
	svm_poly_reg = SVR (kernel='poly', degree=2, C=100, epsilon=0.1)
	svm_poly_reg.fit(X,y)
	#  ‘auto’, uses 1 / n_features.
	svm_rbf_reg = SVR (kernel='rbf', gamma = 'auto', C=100, epsilon=0.1)
	svm_rbf_reg.fit(X,y)
	print('Plotting Learning Curve....\n')
	plot_learning_curves(svm_rbf_reg, svm_poly_reg, X,y)
	return [svm_reg, svm_poly_reg,svm_rbf_reg]


def train_test_df(df):
	"""
	"""
	msk = np.random.rand(len(df)) < 0.8
	return df[msk],df[~msk] 

def ROC_Curve_proba(rf, auc,X_train,X_test,y_train,y_test):
	""" ROC for classifier that outputs probabilities NO SVM
	""" 
	from sklearn.preprocessing import OneHotEncoder

	figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/figurescv5/'
	figures_dir = '/Users/jaime/github/papers/buschke_integral/figures/'
	one_hot_encoder = OneHotEncoder()
	pdb.set_trace()
	rf_fit = rf.fit(X_train, y_train)
	#fit = one_hot_encoder.fit(rf.apply(X_train))
	y_predicted = rf_fit.predict_proba(X_test)[:, 1]
	false_positive, true_positive, _ = roc_curve(y_test, y_predicted)

	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(false_positive, true_positive, color='darkorange', label='Random Forest')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	#plt.title('ROC curve (area = %0.2f)' % auc)
	plt.legend(loc='best')
	figname='roc_curve.png'
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')
	plt.show()

def classification_metrics(df, clf, xlabel, ylabel):
	"""
	"""

	from sklearn.metrics import classification_report, hamming_loss, \
	confusion_matrix, roc_curve, auc
	#target_names = ['H', 'MCI', 'AD']
	target_names = ['H', 'MCI+AD']
	#dfcopy = df.copy()
	#df.loc[df['dx_corto_visita6'] >0, 'dx_corto_visita6'] =1
	y_pred = clf.predict(df[xlabel])
	y_true = df[ylabel]
	print(classification_report(y_true, y_pred, target_names=target_names))
	cm =confusion_matrix(y_true, y_pred)
	auc = roc_auc_score(y_true, y_pred)
	return [cm, auc]

def regression_metrics(df_test, regr, xlabel, ylabel):
	"""
	"""

	from sklearn.metrics import r2_score, accuracy_score,mean_absolute_error,explained_variance_score,max_error

	y_pred = regr.predict(df_test[xlabel])
	y_true = df_test[ylabel]
	me = max_error(y_true, y_pred)
	ev = explained_variance_score(y_true, y_pred)
	r2 = r2_score(y_true, y_pred)
	#The best possible score is 1.0, lower values are worse.
	return [me, ev, r2]

def plot_cm_(cm, clf,X_test, y_test, title):
	"""
	"""
	from sklearn.metrics import plot_confusion_matrix
	# plot CM
	fig, ax = plt.subplots(figsize=(6,6))
	class_names = ["H", "MCI+AD"]
	disp = plot_confusion_matrix(clf, X_test, y_test,values_format='.0f',display_labels=class_names,cmap=plt.cm.Blues)
	disp.ax_.set_title(title)
	print(title)
	print(disp.confusion_matrix)
	fig_png = os.path.join(figures_dir, title + '.png')
	plt.savefig(fig_png)

def tune_hyperparameters_RBF_SVC(X,y):
	"""
	"""
	from sklearn.svm import SVC
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	import itertools

	gammas = [0.001, 0.1, 1, 10]
	Cs = [10, 100, 1000]
	hyperparams = list(itertools.product(gammas, Cs))
	#hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
	svm_clfs = []
	for gamma, C in hyperparams:
		rbf_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))])
		rbf_kernel_svm_clf.fit(X, y)
		svm_clfs.append(rbf_kernel_svm_clf)
	return svm_clfs



def gridsearch_hyperparameter_SVM(X_train, y_train, X_test, y_test):
	"""
	"""
	from sklearn.model_selection import GridSearchCV
	from sklearn.svm import SVC
	from sklearn.metrics import classification_report
	# Set the parameters by cross-validation
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],'C': [1, 10, 100, 1000]},\
	{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},{'kernel':['poly'],'gamma': [1e-2,1e-3, 1e-4],'C': [1, 10, 100, 1000]}]

	scores = ['precision', 'recall', 'f1']
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()

		clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s_macro' % score)
		clf.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		print()
	return clf

def cross_validation_metrics(clf, X,y,cv=5):
	"""compute x validation metrics for a dataset and classifier
	X must be train dataset, do not pass 
	https://scikit-learn.org/stable/modules/cross_validation.html
	"""
	from sklearn.model_selection import cross_val_score
	#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
	scores_list = []
	# cv = 5 by default, cv = int (Stratified)kfold
	scores_acc0 = cross_val_score(clf, X, y, scoring='accuracy')
	scores_list.append(scores_acc0)
	print("Accuracy0: %0.2f (+/- %0.2f)" % (scores_acc0.mean(), scores_acc0.std() * 2))
	scores_acc = cross_val_score(clf, X, y, scoring='balanced_accuracy')
	scores_list.append(scores_acc)
	print("Balanced Accuracy: %0.2f (+/- %0.2f)" % (scores_acc.mean(), scores_acc.std() * 2))
	scores_f1 = cross_val_score(clf, X, y, scoring='f1_macro')
	scores_list.append(scores_f1)
	print("F1 (for unbalanced samples): %0.2f (+/- %0.2f)" % (scores_f1.mean(), scores_f1.std() * 2))
	return scores_list

def plot_bivariate_dataset(X, y, axes, title=None):
	"""X 2 columns
	https://github.com/ageron/handson-ml2/blob/master/05_support_vector_machines.ipynb
	"""
	
	plt.plot(X.values[:, 0][y.values==0], X.values[:, 1][y.values==0], "bs")
	plt.plot(X.values[:, 0][y.values==1], X.values[:, 1][y.values==1], "g^")
	plt.axis(axes)
	plt.grid(True, which='both')
	title = title + '_ '+ X.columns[0]+'_'+X.columns[1]+'__'+y.name
	plt.title(title)
	plt.xlabel(X.columns[0], fontsize=12)
	plt.ylabel(X.columns[1], fontsize=12, rotation=90)
	figname = 'bivariate_'+title+'.png'
	fig_png = os.path.join(figures_dir,figname)
	plt.savefig(fig_png)

def plot_predictions_clf(clf, axes):
	"""
	"""
	
	fig, ax = plt.subplots(figsize=(6,6))
	x0s = np.linspace(axes[0], axes[1], 100)
	x1s = np.linspace(axes[2], axes[3], 100)
	x0, x1 = np.meshgrid(x0s, x1s)
	X = np.c_[x0.ravel(), x1.ravel()]
	y_pred = clf.predict(X).reshape(x0.shape)
	y_decision = clf.decision_function(X).reshape(x0.shape)
	plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
	plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

def plot_bivariate_mind_brain_freesurfer(df_longint):
		# Plot BIVARIATRE MIND-BRAIN
	
	df_longint = atrophy_freesurfer(df_longint)
	print('CALLING TO atrophyLong_paper.py::scatterplot_2variables_in_df bivariate sns ...\n')
	xvar = 'bussum_diff'; yvar = 'hippoL_y6y1'
	scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'BrainSegVol_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'lhCortexVol_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'rhCortexVol_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'CerebralWhiteMatterVol_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'TotalGrayVol_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'SubCortGrayVol_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)

	yvar = 'CC_Central_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'CC_Anterior_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'CC_Posterior_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'hippoL_y6y1'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'CSF_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'CerebralWhiteMatterVol_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'CortexVol_y1_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	#yvar = 'EstimatedTotalIntraCranialVol_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)

	yvar = 'Accumbens_Left_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'Amygdala_Left_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'Caudate_Left_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'Inf_Lat_Vent_Left_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'Putamen_Left_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'Pallidum_Left_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'Thalamus_Proper_Left_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)
	yvar = 'Thalamus_Proper_Right_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir)

	yvar = 'fr_L_thick_G_temporal_middle_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir) 
	yvar = 'fr_R_thick_G_temporal_middle_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir) 
	yvar = 'fr_L_thick_G_precuneus_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir) 
	yvar = 'fr_R_thick_G_precuneus_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir) 
	yvar = 'fr_L_thick_Pole_temporal_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir) 
	yvar = 'fr_R_thick_G_precuneus_diff'; scatterplot_2variables_in_df(df_longint, xvar, yvar, figures_dir) 


def correct_df_types(df, objectlist=None):
	"""correct_df_types convert object type to float
	"""
	for col in objectlist:
		df[col] = df[col].str.replace(',', ".").astype(float)
	return df


def classification_bus_mci(df):
	"""classification_bus_mci study whether B (or) is a good predictor for MCI
	1. Study association B and MCI
		Is there a Gain between B (new B variable) and b (old B variables)?
	2. Build MLP Input Bus + Age, SCD etc Output: MCI 
	"""
	# CSV for classification do not need Brain volumetry
	print('Running classification_bus_mci ....  \n')
	df_orig = df.copy()
	print('Dataframe (rows columns) = (%d %d) \n' %(df.shape[0], df.shape[1]))
	df = correct_df_types(df, ['edad_visita1','edad_ultimodx','sue_noc','imc'])
	print(df.describe())
	
	colsofinterest = ['idpv', 'sexo','edad_visita1', 'lat_manual','edad_ultimodx',\
	'apoe', 'nivel_educativo','anos_escolaridad','familial_ad','dx_corto_visita1',\
	'dx_corto_visita5','mmse_visita1','cn_visita1','animales_visita1',\
	'depre_num','sue_noc','imc', 'dx_corto_visita5','fcsrtrl1_visita1', \
	'fcsrtrl2_visita1', 'fcsrtrl3_visita1','fcsrtrl1_visita5', \
	'fcsrtrl2_visita5', 'fcsrtrl3_visita5','scd_visita1','scd_visita5']
	df = df[colsofinterest]
	print(df.dtypes)
	# Select only those with visit n, just remove rows that have visitn nan
	# nonnans also remove rows with Nans eg apoe or lat manual
	nonnans = lambda dataframe: df[df.notna().all(axis=1)]
	df_1n_ = nonnans(df)
	# What we want is remove rows thta do not have visit n
	df_1n = df[df.fcsrtrl3_visita5.notnull()]
	#df_1n.equals(df_long)
	# df_1n_ = df[df.fcsrtrl1_visita5.notnull()] df_1n__ = df[df.fcsrtrl2_visita5.notnull()]; df_1n.equals(df_1n_); df_1n.equals(df_1n__)
	df_diffs = pd.concat([df_1n, df_1n_]).drop_duplicates(keep=False)
	# nit the same because some nans in apoe 
	print('\n Dataframe of Interest (rows columns) = (%d %d) \n' %(df_1n.shape[0], df_1n.shape[1]))
	print(df_1n.describe())
	print(' Calling to compute_buschke_integral... It will add (5x2) columns for y1 and yn: bus_int_visita1, bus_parint1_visita1,bus_parint2_visita1, bus_sum_visita1, bus_meana_visita1')

	df_1n = compute_buschke_integral(df_1n, 5)
	df_1n['bus_sum_visita1'] = df_1n['bus_sum_visita1'].astype(float)
	# bus_sum_visita1 =  + + 
	# w_0*bus_int_visita1(integral) + w1*'bus_parint1_visita1' + w2*'bus_parint2_visita1 
	# My new variable B.  Integral, weights assume to be 1.0
	weights = np.array([1.0,1.0,1.0])
	df_1n['bus_Integral_visita1'] = weights[0]*df_1n['bus_int_visita1'] + \
	weights[1]*df_1n['bus_parint1_visita1'] + \
	weights[2]*df_1n['bus_parint2_visita1']
	df_1n['bus_Integral_visita1'].iloc[4]
	df_1n[['bus_sum_visita1','bus_Integral_visita1']].describe()
	df_1n[['bus_sum_visita1','bus_Integral_visita1']].idxmax()
	df_1n[['bus_sum_visita1','bus_Integral_visita1']].idxmin() #97 mci year 1 Ad year 5 edad_visita1 72.32
	# Plot distros and calculate correlations sum versus Integral ~ dx_corto_v5
	



def main():
	""" Buschke paper that build a new B. extt based score and we study the associatiob
	of this new variable with conversion to MCI (classification) and Regression Brain tisssue volumetry/atrophy
	"""
	np.random.seed(42)
	global figures_dir
	figures_dir = '/Users/jaime/github/papers/buschke_integral/figures/'

	##################################################
	#### CLASSIFICATION ##############################
	# Rename the file csv_path = "/Users/jaime/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-Siena-15October2019.csv"
	csv_path = "/Users/jaime/github/papers/buschke_integral/datum/for_classification_fsl_1234567_Siena_15102019.csv"
	df = pd.read_csv(csv_path, sep=';', decimal='.') 

	classification_bus_mci(df)
	##################################################


	##################################################
	#### REGRESSION ##################################
	##################################################



	# importlib.reload(JADr_paper); import atrophy_long; atrophy_long.main()
	print('Code for Neural Correlates of Bushcke Memory (immediate)  \n')
	plt.close('all')
	#csv_path = '/Users/jaime/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-Siena-15October2019.csv'
	
	csv_path = "/Users/jaime/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-Siena-Free-27Jan-2019.csv"
	csv_path = "/Users/jaime/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols16-Siena-Free-30012020_segmentation_long.csv"
	# For Regression Hipp y1-y6
	csv_path = "/Users/jaime/vallecas/data/BBDD_vallecas/Free_Longitudinal_y1y6_28Jan2020.csv"
	# For Classification
	csv_path = "/Users/jaime/vallecas/data/BBDD_vallecas/Vols1234567-Siena-Free-23Jan2020_segmentation.csv"
	csv_path = "/Users/jaime/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols1234567-Siena-15October2019.csv"
	dataframe = pd.read_csv(csv_path, sep=';', decimal='.') 
	#dataframe = pd.read_csv(csv_path, sep=';', converters={'y': lambda x: float(x.replace('.','').replace(',','.'))})
	# Copy dataframe with the cosmetic changes e.g. Tiempo is now tiempo
	dataframe_orig = dataframe.copy()
	
	# remove outliers low, high 0.01, 0.99
	#cols = ['R_Hipp_visita1', 'L_Hipp_visita1', 'R_Hipp_visita6', 'L_Hipp_visita6']
	#dataframe = remove_outliers(dataframe, cols)

	#dataframe = convert_stringtofloat(dataframe)
	#dataframe = clean_df(dataframe)
	# Select columns
	colsofinterest = ['idpv', 'sexo','edad_visita1', 'apoe', 'dx_corto_visita6','fcsrtrl1_visita1', \
	'fcsrtrl2_visita1', 'fcsrtrl3_visita1','fcsrtrl1_visita6', \
	'fcsrtrl2_visita6', 'fcsrtrl3_visita6',\
	'R_Hipp_visita1','L_Hipp_visita1','R_Hipp_visita6','L_Hipp_visita6']
	# colsofinterest2 = ['fr_BrainSegVol_y6', 'fr_BrainSegVol_y1','fr_BrainSegVol_to_eTIV_y6','fr_BrainSegVol_to_eTIV_y1','fr_Brain_Stem_y6','fr_Brain_Stem_y1','fr_CC_Central_y6','fr_CC_Central_y1','fr_CC_Anterior_y6','fr_CC_Anterior_y1','fr_CC_Posterior_y6','fr_CC_Posterior_y1','fr_CSF_y6','fr_CSF_y1','fr_CerebralWhiteMatterVol_y6','fr_CerebralWhiteMatterVol_y1','fr_CortexVol_y6','fr_CortexVol_y1','fr_EstimatedTotalIntraCranialVol_y6','fr_EstimatedTotalIntraCranialVol_y1',\
	# 'fr_Left_Accumbens_area_y6','fr_Left_Accumbens_area_y1', 'fr_Left_Amygdala_y6','fr_Left_Amygdala_y1',\
	# 'fr_Right_Amygdala_y6','fr_Right_Amygdala_y1','fr_Left_Caudate_y1','fr_Left_Caudate_y6','fr_Left_Cerebellum_Cortex_y1','fr_Left_Cerebellum_Cortex_y6',\
	# 'fr_Left_Inf_Lat_Vent_y1','fr_Left_Inf_Lat_Vent_y6','fr_Left_Lateral_Ventricle_y1','fr_Left_Lateral_Ventricle_y1',\
	# 'fr_Left_Pallidum_y1','fr_Left_Pallidum_y6','fr_Left_Putamen_y1','fr_Left_Putamen_y6','fr_Left_Thalamus_Proper_y1','fr_Left_Thalamus_Proper_y6',\
	# 'fr_Right_Accumbens_area_y1','fr_Right_Accumbens_area_y6','fr_lhCortexVol_y6','fr_lhCortexVol_y1','fr_rhCortexVol_y6','fr_rhCortexVol_y1',\
	# 'fr_TotalGrayVol_y6','fr_TotalGrayVol_y1','fr_SubCortGrayVol_y6','fr_SubCortGrayVol_y1',\
	# 'fr_Right_Caudate_y6','fr_Right_Caudate_y1','fr_Right_Pallidum_y6','fr_Right_Pallidum_y1','fr_Right_Putamen_y6','fr_Right_Putamen_y1',\
	# 'fr_Right_Thalamus_Proper_y6','fr_Right_Thalamus_Proper_y1','fr_L_thick_G_temporal_middle_y6','fr_L_thick_G_temporal_middle_y1',\
	# 'fr_R_thick_G_temporal_middle_y6','fr_R_thick_G_temporal_middle_y1','fr_L_thick_G_precuneus_y6','fr_L_thick_G_precuneus_y1',\
	# 'fr_R_thick_G_precuneus_y6','fr_R_thick_G_precuneus_y1','fr_L_thick_Pole_temporal_y6','fr_L_thick_Pole_temporal_y1',\
	# 'fr_R_thick_Pole_temporal_y6','fr_R_thick_Pole_temporal_y1']
	# colsofinterest = colsofinterest + colsofinterest2
	
	# colsofinterest for classification y1-y5 
	colsofinterest = ['idpv', 'sexo','edad_visita1', 'apoe', 'dx_corto_visita5','fcsrtrl1_visita1', \
	'fcsrtrl2_visita1', 'fcsrtrl3_visita1','fcsrtrl1_visita5', \
	'fcsrtrl2_visita5', 'fcsrtrl3_visita5']

	colsofinterest_reg = ['dx_corto_visita6','fcsrtrl1_visita6', 'fcsrtrl2_visita6', 'fcsrtrl3_visita6','R_Hipp_visita1','L_Hipp_visita1','R_Hipp_visita6','L_Hipp_visita6']
	colsofinterest = colsofinterest + colsofinterest_reg
	dataframe = dataframe[colsofinterest]
	dataframe[colsofinterest_reg] = dataframe[colsofinterest_reg]/1000.0
	
	dataframe['edad_visita1'] = dataframe['edad_visita1'].str.replace(',', ".").astype(float)


	nonnans = lambda dataframe: dataframe[dataframe.notna().all(axis=1)]
	df_long = nonnans(dataframe)

	df_longint = compute_buschke_integral(df_long, 6)

	# Compare df_longint['bus_int_visita1'] + df_longint['bus_parint1_visita1'] + df_longint['bus_parint2_visita1']
	# with df_longint['bus_sum_visita1']
	#cols2plot = [['bus_parint1_visita1','bus_parint1_visita5'],['bus_parint2_visita1','bus_parint2_visita5'],['bus_sum_visita1','bus_sum_visita5']]
	#plot_Bus_distros(df_longint, cols2plot)

	# Added variables bus_parint1_visita%i ,bus_parint2_visita%i,bus_sum_visita%i, bus_meana_visita%i
	# Add atrophy columns
	# df_longint['hippoL_y6y1'] = df_long['L_Hipp_visita6'] - df_long['L_Hipp_visita1'] 
	# df_longint['hippoR_y6y1'] = df_long['R_Hipp_visita6'] - df_long['R_Hipp_visita1'] 
	# df_longint['bus_int_y6y1'] = df_long['bus_int_visita6'] - df_long['bus_int_visita1'] 
	# df_longint[['hippoL_y6y1','hippoR_y6y1','bus_int_y6y1']].describe() # 'fcsrtrl1_visita1', 'fcsrtrl2_visita1', 'fcsrtrl3_visita1', 'bus_int_visita1','bus_sum_visita1','bus_meana_visita1','R_Hipp_visita1','L_Hipp_visita1','R_Hipp_visita6','L_Hipp_visita6']].describe()
	# df_longint.dropna(subset=['hippoL_y6y1', 'hippoR_y6y1','bus_int_y6y1'], how='any', inplace=True)
	
	# Normalize
	# df_longint['hippoL_y6y1']-= df_longint['hippoL_y6y1'].min() 
	# df_longint['hippoL_y6y1']/= df_longint['hippoL_y6y1'].max() 
	
	#cols2plotlist=[['hippoL_y6y1'],['hippoR_y6y1']]
	#plot_brain_distros(df_longint,cols2plotlist)

	# B Integral sum of integral13 + partial integral12 and partial integral23 (weights assume to be 1)
	w = np.array([1.0,1.0,1.0])
	df_longint['bus_Integral_visita1'] = w[0]*df_longint['bus_int_visita1'] + w[1]*df_longint['bus_parint1_visita1'] + w[2]*df_longint['bus_parint2_visita1'] 
	# df_longint['bus_Integral_visita6'] = w[0]*df_longint['bus_int_visita6'] + w[1]*df_longint['bus_parint1_visita6'] + w[2]*df_longint['bus_parint2_visita6'] 
	# df_longint['bussum_diff'] = df_longint['bus_sum_visita6'].sub(df_longint['bus_sum_visita1'], axis = 0)
	# # bus_int_visita6 the integral between points 1 and 3 (not == sum)
	# df_longint['bus_intsum_diff'] = df_longint['bus_int_visita6'].sub(df_longint['bus_int_visita1'], axis = 0)
	# df_longint['bus1_diff'] = df_longint['bus_parint1_visita6'].sub(df_longint['bus_parint1_visita1'], axis = 0)
	# df_longint['bus2_diff'] = df_longint['bus_parint2_visita6'].sub(df_longint['bus_parint2_visita1'], axis = 0)

	# df_longint['bus_Integral_diff'] = df_longint['bus_Integral_visita6'].sub(df_longint['bus_Integral_visita1'], axis = 0)
	
	# DX_S = distance_correlation(df_longint['hippoL_y6y1'],df_longint['bussum_diff'] )
	# DX_1 = distance_correlation(df_longint['hippoL_y6y1'],df_longint['bus1_diff'] )
	# DX_2 = distance_correlation(df_longint['hippoL_y6y1'],df_longint['bus2_diff'] )
	# DX_Int = distance_correlation(df_longint['hippoL_y6y1'],df_longint['bus_Integral_diff'] )
	
	# xvar = 'bus_int_y6y1'; yvar = 'hippoL_y6y1'
	# print('Plotting Scatter plot %s and %s' %(yvar, xvar))
	# OLS_regression(df_longint)

	#plot_bivariate_mind_brain_freesurfer(df_longint)
	


	# Classifier
	# Compare bus_Integral_visita1 with bus_sum_visita1 or 
	# the differences y6y1 bus_Integral_diff bussum_diff
	df_healthy =  df_longint.loc[df_longint['dx_corto_visita5'] <1]
	df_longint_multiclass = df_longint.copy()
	df_longint.loc[df_longint['dx_corto_visita5'] >0, 'dx_corto_visita5'] =1

	df_train, df_test = train_test_df(df_longint)

	#[svm_clf,poly_svm_clf,rbf_svm_clf] = svm_classifier(df_train, ['Accumbens_Left_diff','Amygdala_Left_diff','Caudate_Left_diff'], 'dx_corto_visita6')
	input_labels = ['edad_visita1', 'bus_Integral_visita1', 'sexo']
	#input_labels = ['sexo']
	#input_labels = ['bus_sum_visita1']

	output_label_clf = 'dx_corto_visita5'; output_label_reg = 'L_Hipp_visita1' #'hippoL_y6y1'
	[svm_clf, poly_svm_clf, rbf_svm_clf] = svm_classifier(df_train, input_labels, output_label_clf)
	# select clf parameters
	gamma = rbf_svm_clf.get_params()['svm_clf__gamma']; C = rbf_svm_clf.get_params()['svm_clf__C']
	#print('\n\n X = TRAIN+TEST \n\n')
	X = df_longint[input_labels]; y = df_longint[output_label_clf]
	xval_metrics = cross_validation_metrics(rbf_svm_clf, X,y)

	#Parameter estimation using grid search with cross-validation
	#X_train = df_train[input_labels];y_train = df_train[output_label_clf];X_test = df_test[input_labels];y_test = df_test[output_label_clf] 
	X = df_longint[input_labels]; y = df_longint[output_label_clf]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	
	# Neural Networks: MLP
	#X = df_longint[output_label_reg].values.reshape(-1,1);
	X = df_longint[['L_Hipp_visita1', 'L_Hipp_visita6']]
	#X = df_longint[['bus_Integral_visita1', 'edad_visita1']]
	y = df_longint['L_Hipp_visita1']
	mlp_reg = MLP_regressor(X, y)
	pdb.set_trace()
	y = df_longint['L_Hipp_visita6']
	mlp_reg = MLP_regressor(X, y)
	pdb.set_trace()
	print('Right....\n')
	y = df_longint['R_Hipp_visita1']
	mlp_reg = MLP_regressor(X, y)
	y = df_longint['R_Hipp_visita6']
	mlp_reg = MLP_regressor(X, y)




	pdb.set_trace()
	mlp_model = MLP_classifier(X, y)
	# Save and Restore the model (previous fit)
	datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
	mlp_model.save("keras_model" + datestring + ".h5")
	#model = keras.models.load_model("my_keras_model.h5")
	pdb.set_trace()

	grid_clf = gridsearch_hyperparameter_SVM(X_train, y_train, X_test, y_test)
	print('SVM best params is %s' %grid_clf.best_params_)
	pdb.set_trace()

	print('Calling to classification_metrics for TRAIN ...\n')
	[cm_tr, auc_tr] = classification_metrics(df_train, rbf_svm_clf, input_labels, output_label_clf)
	print('Calling to classification_metrics for TEST ...\n')
	[cm_tt, auc_tt] = classification_metrics(df_test, rbf_svm_clf, input_labels, output_label_clf)
	
	X_train = df_train[input_labels]; X_test = df_test[input_labels]
	y_train = df_train[output_label_clf]; y_test = df_test[output_label_clf]
	
	#ROC_Curve_proba(svm_clf, auc, X_train,X_test,y_train,y_test)
	#scatterplot_2variables_in_df(df_train, ['bus_Integral_diff','edad_visita1'], 'hippoL_y6y1', figures_dir)
	title = ['CM train svm_clf','CM train poly_svm_clf','CM train rbf_svm_clf']
	#plot_cm_(cm, svm_clf, X_train, y_train, title[0]);plot_cm_(cm, poly_svm_clf, X_train, y_train, title[1]);
	plot_cm_(cm_tr, rbf_svm_clf, X_train, y_train, title[2]);
	title = ['CM test svm_clf','CM test poly_svm_clf','CM test rbf_svm_clf']
	#plot_cm_(cm, svm_clf, X_test, y_test, title[0]);plot_cm_(cm, poly_svm_clf, X_test, y_test, title[1]);
	plot_cm_(cm_tt, rbf_svm_clf, X_test, y_test, title[2]);


	# Plot predictions SVM classifier
	axesx = [df_longint['bus_Integral_visita1'].min(),df_longint['bus_Integral_visita1'].max()]; 
	axesy = [df_longint['edad_visita1'].min(),df_longint['edad_visita1'].max()]	
	plot_predictions_clf(rbf_svm_clf, [axesx[0], axesx[1],axesy[0], axesy[1]])

	kernel = rbf_svm_clf.get_params()['svm_clf__kernel']+str(gamma)+'C'+str(C)
	plot_bivariate_dataset(df_longint[['bus_Integral_visita1','edad_visita1']], df_longint['dx_corto_visita6'], [axesx[0], axesx[1],axesy[0], axesy[1]],kernel)
	#save_fig("moons_polynomial_svc_plot")
	
	# Find best hyprparamters RBF SVM
	X = df_train[input_labels]; y = df_train[output_label_clf]
	svm_clfs = tune_hyperparameters_RBF_SVC(X, y)
	for clf in svm_clfs:
		gamma = clf.get_params()['svm_clf__gamma']; C = clf.get_params()['svm_clf__C']
		print('Classification Training Metrics for SVM params: Gamma %s C %s' %(gamma, C))
		[cm_tr, auc_tr] = classification_metrics(df_train, clf, input_labels, output_label_clf)
		print('Classification TEST Metrics for SVM params: Gamma %s C %s' %(gamma, C))
		[cm_0, auc_0] = classification_metrics(df_test, clf, input_labels, output_label_clf)
		# plot Decision Boundaries
		plot_predictions_clf(clf, [axesx[0], axesx[1], axesy[0], axesy[1]])
		kernel = str(clf.get_params()['svm_clf__kernel'])+'G'+str(gamma)+'C'+str(C)
		plot_bivariate_dataset(df_longint[['bus_Integral_visita1','edad_visita1']], df_longint['dx_corto_visita6'], [axesx[0], axesx[1],axesy[0], axesy[1]],kernel)
	# svm_clfs[1] gamma = 0.1 C=1000 better results
	pdb.set_trace()


	# Regression
	[svm_reg, svm_poly_reg, svm_rbf_reg] = svm_regression(df_train, input_labels, output_label_reg) 
	pdb.set_trace()
	[a,b,c] = regression_metrics(df_test, svm_reg, input_labels, output_label_reg)

	[a,b,c] = regression_metrics(df_test, svm_reg, ['bus_Integral_diff','sexo','edad_visita1', 'apoe'], 'hippoR_y6y1')
	[a,b,c] = regression_metrics(df_test, svm_reg, ['bus_Integral_diff','sexo','edad_visita1', 'apoe'], 'CSF_diff')
	[a,b,c] = regression_metrics(df_test, svm_reg, ['bus_Integral_diff','sexo','edad_visita1', 'apoe'], 'TotalGrayVol_diff') 
	pdb.set_trace()
	


	X = pd.DataFrame(df_longint['bus_int_y6y1']); y = pd.DataFrame(df_longint['hippoL_y6y1'])

	regression_model = regression_mv(X, y)
	dcorHL = distcorr(df_longint['bus_int_y6y1'], df_longint['hippoL_y6y1'])
	dcorHR = distcorr(df_longint['bus_int_y6y1'], df_longint['hippoR_y6y1'])
	pdb.set_trace()

	
