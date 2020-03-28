# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:22:22 2017

@author: Admin
"""

# -*- coding: utf-8 -*-


def LogisticReg_KS_add(df_train, df_test, target, para_list):
    import statsmodels.api as sm        
    ############# re-train the model #####################
    logit = sm.Logit(df_train[target],df_train[para_list])
    result_temp = logit.fit()
    p = result_temp.summary2()
    p_value = p.tables[1][u'P>|z|'][-1]
#    result_temp.summary()
#    result_temp.params
    ##################### get score for training set ######################
    df_train['prob_bad'] = result_temp.predict(df_train[para_list])
    ks_train = ks_group(df_train,target, 'prob_bad', 20, True).ks.max()
    #####################################################################
    df_test['prob_bad'] = result_temp.predict(df_test[para_list])
    ks_test = ks_group(df_test,target, 'prob_bad', 20, True).ks.max()
    return [ks_train,ks_test,p_value]



def Marginal_Add(df_train,df_test,target,model_para_list):
    [ks_train,ks_test,p_value]= LogisticReg_KS_add(df_train,df_test,target, model_para_list)
    result = pd.DataFrame({"var_name": 'original', "KS_train": ks_train, "KS_test": ks_test, "P_Value": p_value}, index=["0"])
    
    all_para_list = df_train.columns.tolist()
    step2_list = [x for x in all_para_list if x not in model_para_list and x not in [target,'apply_id']]

    for var in step2_list:
        model_para_list_add = model_para_list + [var]
        [ks_train,ks_test,p_value]= LogisticReg_KS_add(df_train,df_test,target, model_para_list_add)
        
#        df_test['prob_bad'] = model_result.predict(df_test[model_para_list_add])
#        ks_test = ks_group(df_test,target, 'prob_bad', 20, True).ks.max()

        result_temp = pd.DataFrame({"var_name": var, "KS_train": ks_train, "KS_test": ks_test, "P_Value": p_value}, index=["1"])
        result = result.append(result_temp)
    result.sort_values('KS_train',ascending=False,inplace=True)
    return result

def LogisticReg_KS_add2(df_train, df_test, target, para_list):
    import statsmodels.api as sm        
    ############# re-train the model #####################
    logit = sm.Logit(df_train[target],df_train[para_list])
    result_temp = logit.fit()
    p = result_temp.summary2()
    p_value = p.tables[1][u'P>|z|'][-1]
#    result_temp.summary()
#    result_temp.params
    ##################### get score for training set ######################
    df_train['prob_bad'] = result_temp.predict(df_train[para_list])
    ks_train = ks_group_equal(df_train,target, 'prob_bad', 20, True).ks.max()
    #####################################################################
    df_test['prob_bad'] = result_temp.predict(df_test[para_list])
    ks_test = ks_group_equal(df_test,target, 'prob_bad', 20, True).ks.max()
    return [ks_train,ks_test,p_value]



def Marginal_Add2(df_train,df_test,target,model_para_list):
    [ks_train,ks_test,p_value]= LogisticReg_KS_add2(df_train,df_test,target, model_para_list)
    result = pd.DataFrame({"var_name": 'original', "KS_train": ks_train, "KS_test": ks_test, "P_Value": p_value}, index=["0"])
    
    all_para_list = df_train.columns.tolist()
    step2_list = [x for x in all_para_list if x not in model_para_list and x not in [target,'apply_id']]

    for var in step2_list:
        model_para_list_add = model_para_list + [var]
        [ks_train,ks_test,p_value]= LogisticReg_KS_add2(df_train,df_test,target, model_para_list_add)
        
#        df_test['prob_bad'] = model_result.predict(df_test[model_para_list_add])
#        ks_test = ks_group(df_test,target, 'prob_bad', 20, True).ks.max()

        result_temp = pd.DataFrame({"var_name": var, "KS_train": ks_train, "KS_test": ks_test, "P_Value": p_value}, index=["1"])
        result = result.append(result_temp)
    result.sort_values('KS_train',ascending=False,inplace=True)
    return result
