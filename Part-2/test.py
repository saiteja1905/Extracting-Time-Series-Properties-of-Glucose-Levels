import pandas as pd
from scipy.fftpack import rfft
import numpy as np
from joblib import load

df=pd.read_csv('test.csv',header=None)


def getnomeal_ftr(no_meal):
    p_f=[]
    index_first=[]
    
    
    i1=no_meal.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    data_t=no_meal.drop(no_meal.index[i1]).reset_index().drop(columns='index')
    data_t1=data_t.interpolate(method='linear',axis=1)
    i2=data_t1.isna().sum(axis=1).replace(0,np.nan).dropna().index
    
    p_s=[]
    index_second=[]
    p_t=[]
    frst_diff_data=[]
    sec_diff_data=[]
    sd=[]
    
    non_meal_data_postprocess=data_t1.drop(data_t1.index[i2]).reset_index().drop(columns='index')
    non_meal_feature_matrix=pd.DataFrame()
    
    l=len(non_meal_data_postprocess)
    
    for i in range(l):
        array=abs(rfft(non_meal_data_postprocess.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(non_meal_data_postprocess.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        p_f.append(sorted_array[-2])
        p_s.append(sorted_array[-3])
        p_t.append(sorted_array[-4])
        index_first.append(array.index(sorted_array[-2]))
        index_second.append(array.index(sorted_array[-3]))

    non_meal_feature_matrix['power_second_max']=p_s
    non_meal_feature_matrix['power_third_max']=p_t

    
    for i in range(l):
        frst_diff_data.append(np.diff(non_meal_data_postprocess.iloc[:,0:24].iloc[i].tolist()).max())
        sec_diff_data.append(np.diff(np.diff(non_meal_data_postprocess.iloc[:,0:24].iloc[i].tolist())).max())
        sd.append(np.std(non_meal_data_postprocess.iloc[i]))

    non_meal_feature_matrix['2ndDifferential']=sec_diff_data
    non_meal_feature_matrix['sd']=sd
    return non_meal_feature_matrix


d=getnomeal_ftr(df)




with open('model.pkl', 'rb') as pre_trained:
    pickle_file = load(pre_trained)
    predict = pickle_file.predict(d)    
    pre_trained.close()



pd.DataFrame(predict).to_csv('Results.csv',index=False,header=False)