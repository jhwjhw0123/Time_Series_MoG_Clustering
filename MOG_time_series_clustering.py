import numpy as np
import math
import random
from numpy import linalg
from numpy import matlib
import os
import pandas as pd
import sklearn
from sklearn import mixture

def time_series_embedding(dic_path,emd_len):
    #Process the data into time series
    all_files = os.listdir(dic_path)
    data_all_files = []
    for file in all_files:
        this_file_name = dic_path+file
        attr_name = file[:-4]   #arribute name
        try:
            This_file = pd.read_csv(this_file_name)
            date_series = This_file[attr_name]
            series_len = date_series.shape[0]
            #Calculate here we could have n-d lags
            nLags = series_len-emd_len
            if(nLags<=0):
                raise ValueError('Embedding length is longer than data series length. Not permitted!')
            data_this_file = []
            for i in range(nLags):
                data_this_file.append(date_series[i:i+emd_len])
            data_this_file = np.asarray(data_this_file)
            data_all_files.append(data_this_file)
        except:
            #Currently 'pool' file cannot be read properly
            print('Meet an error in reading'+this_file_name+', has been ignored...')

    return data_all_files

def log_likelihood_MoG(data,mean,covariance,cate_para):
    #*****************************
    #   data should be nData * nDim
    #   mean should be a numpy array with nGaussian * nDim
    #   Covariance should be a numpy array with nGaussian * (nDim * nDim)
    #   Cata_para is the parameter for the categorical distribution, should be nGaussian * 1 vector
    nData = data.shape[0]
    nDim = data.shape[1]
    nGassian = mean.shape[0]
    marginal_likelihood = []
    #try to use vectorized method to compute as possible
    for cGaussian in range(nGassian):
        this_mean = np.reshape(mean[cGaussian],[nDim,1])
        this_cov = covariance[cGaussian]
        this_cate = cate_para[cGaussian]
        inv_cov = linalg.inv(this_cov)
        cent_data = data - (matlib.repmat(this_mean,1,nData)).T
        exp_term = -0.5*np.diag(np.matmul(np.matmul(cent_data,inv_cov),cent_data.T))
        this_likelihood = this_cate * math.pow((2*math.pi),(-nDim/2))*math.pow(linalg.det(this_cov),-0.5)*np.exp(exp_term)
        #print(this_likelihood.shape)
        marginal_likelihood.append(this_likelihood)

    marginal_likelihood = np.asarray(marginal_likelihood)      #nGaussian * nData
    sum_marg_likelihood = np.sum(marginal_likelihood,axis=0)
    log_sum_likelihood = np.log(sum_marg_likelihood)
    final_log_likelihood = np.sum(log_sum_likelihood)/nData

    return final_log_likelihood

def variation_dis_MoG(data,mean,covariance,cate_para):
    # *****************************
    #   data should be nData * nDim
    #   mean should be a numpy array with nGaussian * nDim
    #   Covariance should be a numpy array with nGaussian * (nDim * nDim)
    #   Cata_para is the parameter for the categorical distribution, should be nGaussian * 1 vector
    nData = data.shape[0]
    nDim = data.shape[1]
    nGassian = mean.shape[0]
    marginal_likelihood = []
    # try to use vectorized method to compute as possible
    for cGaussian in range(nGassian):
        this_mean = np.reshape(mean[cGaussian], [nDim, 1])
        this_cov = covariance[cGaussian]
        this_cate = cate_para[cGaussian]
        inv_cov = linalg.inv(this_cov)
        cent_data = data - (matlib.repmat(this_mean, 1, nData)).T
        exp_term = -0.5 * np.diag(np.matmul(np.matmul(cent_data, inv_cov), cent_data.T))
        this_likelihood = this_cate * math.pow((2 * math.pi), (-nDim / 2)) * math.pow(linalg.det(this_cov),
                                                                                          -0.5) * np.exp(exp_term)
        # print(this_likelihood.shape)
        marginal_likelihood.append(this_likelihood)

    marginal_likelihood = np.asarray(marginal_likelihood).T  # nData * nGaussian
    sum_likelihood = matlib.repmat(np.reshape(np.sum(marginal_likelihood,axis=1),[nData,1]),1,nGassian)
    variational_q_distribution = np.divide(marginal_likelihood,sum_likelihood)


    return variational_q_distribution


def time_series_MoG(data,nGaussian,max_Int=100,tol=0.2):
    #***************************************
    #   Author: Chen Wang, UCL Dept. of Computer Science
    #   This is the Mixture of Gaussian method specified for time series data
    #   The general concept to train this is still to use E-M algorithm, however there are special constrains
    #   because the data is not i.i.d
    #   This is based on the paper from Eirola et al. in 2013, titled:
    #   "Gaussian Mixture Models for Time Series Modelling, Forecasting, and Interpolation"
    nData = data.shape[0]
    nDim = data.shape[1]
    # nGaussian * nDim mean vector
    mean_MoG = np.random.normal(loc=0.0,scale=1.0,size=[nGaussian,nDim])
    # nGaussian * (nDim * nDim) list of covariance
    # here we use this routine to guarantee positive definite
    cov_MoG = []
    for cGaussian in range(nGaussian):
        random_mat = np.random.normal(loc=0.0,scale=1.0,size=[nDim,nDim])
        this_cov = np.matmul(random_mat,random_mat.T)+random.random()*np.eye(nDim)
        cov_MoG.append(this_cov)
    cov_MoG = np.asarray(cov_MoG)
    # parameter for the categorical distribution, here we initialize with equal probability distribution
    cate_para = (1/nGaussian)*np.ones([nGaussian])
    prev_log_likelihood = 1
    log_likelihood = -1000
    for ite in range(max_Int):
        log_likelihood = log_likelihood_MoG(data,mean_MoG,cov_MoG,cate_para)
        print('The',ite+1,'iteration, the log-likelihood is:',log_likelihood)
        if(math.isnan(log_likelihood)==True):
            raise ValueError('Numerical Unstable..try again')
        if(abs((log_likelihood-prev_log_likelihood)/prev_log_likelihood)<=tol):
            print('The iteration has converged and terminated')
            break
        elif(ite==(max_Int-1)):
            print('Maximum iteration time has been achieved. Terminated without convergence...')
        #E-step, computing responsibility
        variation_q_dis = variation_dis_MoG(data,mean_MoG,cov_MoG,cate_para)    #nData * nGaussian
        total_variation_q = np.sum(variation_q_dis,axis=0)                      #sum among all the data
        #M-step, update the parameters
        for cGaussian in range(nGaussian):
            N_cGaussian = np.sum(variation_q_dis[:,cGaussian])
            #Compute the mean parameter
            x_with_coeff = np.multiply(matlib.repmat(np.reshape(variation_q_dis[:,cGaussian],[nData,1]),1,nDim),data)
            this_mean = np.sum(x_with_coeff,axis=0)/N_cGaussian
            #Compute the covariance parameter
            cent_data = data - (matlib.repmat(np.reshape(this_mean,[nDim,1]), 1, nData)).T
            cent_data_coeff = np.multiply(matlib.repmat(np.reshape(variation_q_dis[:,cGaussian],[nData,1]),1,nDim),cent_data)
            this_cov = np.matmul(cent_data_coeff.T,cent_data)/N_cGaussian
            #Comnpute the categorical parameter
            this_cate_para = total_variation_q[cGaussian]/np.sum(total_variation_q)
            #Update all three parameters
            mean_MoG[cGaussian]  = this_mean
            cov_MoG[cGaussian] = this_cov
            cate_para[cGaussian] = this_cate_para
        prev_log_likelihood = log_likelihood
        # special process for time series data (based on the paper)
        # calculate the global mean with \sigma(lambda_k*u_k)
        global_mean = np.sum(np.multiply(mean_MoG,matlib.repmat(np.reshape(cate_para,[nGaussian,1]),1,nDim)),axis=0)
        # Then calculate the mean value of the vector and centralize global mean
        m_value = np.mean(global_mean)
        # compute the delta value in the paper
        global_mean_central = global_mean - m_value*np.ones([nDim])
        rep_delta = matlib.repmat(np.reshape(global_mean_central,[nDim,1]).T,nGaussian,1)
        # update mean value
        org_mean = mean_MoG
        mean_MoG = mean_MoG - np.multiply(matlib.repmat(np.reshape(cate_para/np.sum(np.power(cate_para,2)),[nGaussian,1]),1,nDim),rep_delta)
        # update convariance value
        for cGaussian in range(nGaussian):
            this_org_mean = np.reshape(org_mean[cGaussian],[nDim,1])
            this_trans_mean = np.reshape(mean_MoG[cGaussian],[nDim,1])
            cov_MoG[cGaussian] = cov_MoG[cGaussian] + np.matmul(this_org_mean,this_org_mean.T) - \
                                 np.matmul(this_trans_mean,this_trans_mean.T)

    return mean_MoG, cov_MoG, cate_para, log_likelihood

def model_selection(data,model='BIC',max_Int=100,max_Gaussian=10):
    nData = data.shape[0]
    nDim = data.shape[1]
    criteria_collection = []
    mean_collection = []
    cov_collection = []
    cate_para_collection = []
    likelihood_collection = []
    for cComponent in range(max_Gaussian):
        nComponent = cComponent+2   #Offset the model to 2~maximum instead of 0~(maximum-1)
        this_likelihood_collection = []
        this_mean_collection = []
        this_cov_collection = []
        this_cate_para_collection = []
        print('Testing MoG with ',nComponent,' components...')
        while len(this_likelihood_collection)<10:  #Every program try 10 times
            try:
                mean_MoG, cov_MoG, cate_para, log_likelihood = time_series_MoG(data,nComponent,max_Int=max_Int)
                this_likelihood_collection.append(log_likelihood)
                this_mean_collection.append(mean_MoG)
                this_cov_collection.append(cov_MoG)
                this_cate_para_collection.append(cate_para)
            except:
                pass
        P = (nComponent-1)*nDim +1 + 0.5*(nComponent-1)*nDim*(nDim+1) + nComponent-1
        this_likelihood_collection = np.asarray(this_likelihood_collection)
        this_select_ind = np.argmax(this_likelihood_collection)
        log_likelihood = this_likelihood_collection[this_select_ind]
        mean_MoG = this_mean_collection[this_select_ind]
        cov_MoG = this_cov_collection[this_select_ind]
        cate_para = this_cate_para_collection[this_select_ind]
        if (model=='AIC'):
            this_cretira_value = -2*log_likelihood + 2*P
        elif (model=='BIC'):
            this_cretira_value = -2*log_likelihood + math.log(nData)*P
        elif (model=='likelihood'):
            this_cretira_value = -2*log_likelihood
        else:
            raise ValueError('The data selection method not recognized! Only admit AIC and BIC!')
        criteria_collection.append(this_cretira_value)
        mean_collection.append(mean_MoG)
        cov_collection.append(cov_MoG)
        cate_para_collection.append(cate_para)
        likelihood_collection.append(log_likelihood)
    criteria_collection = np.asarray(criteria_collection)
    opt_component_ind =  np.argmin(criteria_collection)
    opt_component_num = opt_component_ind + 2
    print('The model model under this data is to use',opt_component_num,'Gaussians')

    opt_mean_MoG = mean_collection[opt_component_ind]
    opt_cov_MoG = cov_collection[opt_component_ind]
    opt_cate_para = cate_para_collection[opt_component_ind]
    opt_likelihood = likelihood_collection[opt_component_ind]

    return opt_mean_MoG, opt_cov_MoG, opt_cate_para, opt_likelihood

dic_path = '../data/univariate_data/'
#Data is collected as a list, with each component as (nLag * embedding_length) array
holistic_data = time_series_embedding(dic_path,12)
#Shape of data
#   mean_MoG: [nGaussian, nDim]
#   cov_MoG: [nGaussian, nDim, nDim]
#   cate_para: [nGaussian,1]
#mean_MoG, cov_MoG, cate_para, log_likelihood = time_series_MoG(holistic_data[0],3,max_Int=100)

mean_MoG, cov_MoG, cate_para, log_likelihood = model_selection(holistic_data[0],model='AIC',max_Int=100)

