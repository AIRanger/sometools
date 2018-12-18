class Rtools():
    # 建模过程：
    # 需要有三个文件夹: datasource、reportsource、figuresource、modelsource



    def __init__(self,name):
        import sys
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl   # mpl.style.available
        mpl.style.use('ggplot')
        from sklearn.metrics import roc_curve, auc
        from xgboost import XGBClassifier
        from sklearn.model_selection import GridSearchCV
        import xgboost as xgb
        from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
        import os
        import pyce
        import time

        # 创建文件夹用于保存过程中产生的数据、报告、图表、模型
        if not os.path.exists('./'+name+'datasource'):
            os.mkdir('./'+name+'datasource')
        if not os.path.exists('./'+name+'reportsource'):
            os.mkdir('./'+name+'reportsource')
        if not os.path.exists('./'+name+'figuresource'):
            os.mkdir('./'+name+'figuresource')
        if not os.path.exists('./'+name+'modelsource'):
            os.mkdir('./'+name+'modelsource')

        self.plt = plt
        self.roc_curve = roc_curve
        self.auc = auc
        self.KFold = KFold
        self.SKFold = StratifiedKFold
        self.train_test_split = train_test_split
        self.pd = pd
        self.XGBClassifier = XGBClassifier
        self.xgbModel = xgb
        self.GSCV = GridSearchCV
        self.features_prof_recode = pyce.features_prof_recode
        self.hive = hive
        self.parms = None




        def fold_cross_validation(self):
            pass

        def get_parms(self,df,parms,oot_timepoint = '2018-11-26',oot_percent = None,,partition_time_var = 'date',label ='label'):
            """
            parms:


            """



            self.parms = parms

            return parms


        def psi(self):# 可做成fit、transfrom形式
            pass

        # def 





        def plot_ks(self,y_true, y_pred, text='', ax=None):
            '''
            parms:
            y_true : 真实标签
            y_pred : 预测的概率值
            '''
            if not ax:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            ks = max(abs(tpr - fpr))
            auc_score = auc(fpr, tpr)
            x = [1.0 * i / len(tpr) for i in range(len(tpr))]
        
            cut_index = (tpr - fpr).argmax()
            cut_x = 1.0 * cut_index / len(tpr)
            cut_tpr = tpr[cut_index]
            cut_fpr = fpr[cut_index]
        # plt.rcdefaults()#重置rc所有参数，初始化
        # plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        # plt.rc('figure', figsize=(8,6))
        # plt.figure(figsize = (8, 6), dpi=100) 
            ax.plot(x, tpr, lw=2, label='TPR')
            ax.plot(x, fpr, lw=2, label='FPR')    
            ax.plot([cut_x,cut_x], [cut_tpr,cut_fpr], color='firebrick', ls='--')  
            ax.text(0.45, 0.3, 'KS = {:.2f},AUC = {:.2f}'.format(ks,auc_score))       
            ax.set_xlabel('Proportion')
            ax.set_ylabel('Rate')
            ax.set_title(f'{text} K-S curve')
            ax.legend(loc="lower right")
            plt.show()


        def ev_plot(y_true,y_pre):

            
            pass
	def __get_points(self,score):
            points = []
            if isinstance(score,list):
                score = self.pd.Series(score)
            interval = round((score.max() - score.min())/10,6)
            for i in range(10):
                points.append(round(score.min()+i*interval,6))
            #         print(int(score.min())+i*interval)
            points.append(round(score.max(),6))
            return points
    
        #
        def __get_points2(self,score,bins_Num):
            #points = []
            #bins=pd.qcut(score, bins_Num)
            #for ix,i in enumerate(bins.value_counts().sort_index().index.values):
            #    points.append(i.right)
            #del points[bins_Num-1]
            points = pd.qcut(score, bins_Num,retbins = True)[1]
        
            del points[0]
            del points[bins_Num-1]
        
            return points
        #       
    
        def __cal_ratio(self,score,points):
            p = []
            for i in range(10):
                p.append(sum((score>=points[i])&(score<points[i+1]))/len(score)+0.00000000000001)
            return p
    
        def __cal_psi(self,p_expect,p_real):
            import math
            psi = 0
            for m,n in zip(p_expect,p_real):
                psi_n = (n-m)*math.log(n/m)
                psi += psi_n

            return psi
    
    def run_psi(self,standard,test_data):
        points = self.__get_points(standard)
        p_base = self.__cal_ratio(standard,points)
        p_test = self.__cal_ratio(test_data,points)
        psi = self.__cal_psi(p_base,p_test)
        return psi

