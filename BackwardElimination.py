# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:14:04 2021

@author: lord of music
"""

def BackwardElimination(X,y):
    import statsmodels.api as stm
    import numpy as np
    
    X=np.append(arr=np.ones((X.shape[0],1)).astype(int),values=X,axis=1)
    ColumnIndexList=[i for i in range(X.shape[1])]
    X_opt=X[:,ColumnIndexList]
    #Step2 : Fit the model with all possible predictors
    regressor_opt=stm.OLS(y,X_opt).fit()
    #Step3:Consider the prdictor with the highest P value if P>sl=0.05 Go to step 4 else Finish 
    pValuesList=list(regressor_opt.pvalues)
   
    while(True):
        verification=False
        for e in pValuesList:
            if e>0.05 :   
                ColumnIndexList.pop(pValuesList.index(e))
                pValuesList.remove(e)
                X_opt=X[:,ColumnIndexList]
                regressor_opt=stm.OLS(y,X_opt).fit()
            else :
                continue
        pValuesList=list(regressor_opt.pvalues)
        for e in pValuesList:
            if e>0.05 :
                verification=True
        if verification==False :
            break 
        else :
            continue
    return(pValuesList,regressor_opt)
