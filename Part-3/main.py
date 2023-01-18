from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import numpy 
import pandas

def BWZCradInput(sample_input1):
    sample_inputData1 = pandas.read_csv(sample_input1,parse_dates=[['Date','Time']])
    df1 = sample_inputData1[['Date_Time', 'BWZ Carb Input (grams)']]
    df1 = df1.rename(columns={'BWZ Carb Input (grams)': 'meal'})
    return df1

def get_Both_Input(sample_input2, sample_input1):
    return sensor_isig_values(sample_input2), BWZCradInput(sample_input1)

def meal_insulin(cdp, insulin_mp):
    df2 = cdp.copy()
    df1 = insulin_mp.copy()
    df2 = df2.loc[df2['sensor'].notna()]
    df2.set_index(['Date_Time'],inplace=True)
    df2 = df2.sort_index()
    df2 = df2.reset_index()
    df1.set_index(["Meal_Time"],inplace=True)
    df1 = df1.sort_index()
    df1 = df1.reset_index()
    r = pandas.merge_asof(df1, df2,left_on='Meal_Time',right_on='Date_Time',direction="forward")
    return r

def meal_feature_matrix(Meal_dataframe, df2):
    inputmeal_data = []
    Meal_dataframe.reset_index()

    for index, x in Meal_dataframe.iterrows():
        meal_begin = x['Date_Time'] + pandas.DateOffset(minutes=-30)
        stop = x['Date_Time'] + pandas.DateOffset(hours=2)
        meal_data = df2.loc[(df2['Date_Time'] >= meal_begin)&(df2['Date_Time']<stop)]
        meal_data.set_index('Date_Time',inplace=True)
        meal_data = meal_data.sort_index()
        meal_data = meal_data.reset_index()
        isCorrect, meal_data = glucose_ts(meal_data, 30)
        if isCorrect == False:
            continue
        meal_data = meal_data[['sensor']]
        array1 = meal_data.to_numpy().reshape(1, 30)
        array1 = numpy.insert(array1, 0, index, axis=1)
        array1 = numpy.insert(array1, 1, x['meal'], axis=1)
        inputmeal_data.append(array1)

    return numpy.array(inputmeal_data).squeeze()

def glucose_ts(dataframe, vc):
    sensor_not_null_values = dataframe.loc[dataframe['sensor'].notna()]['sensor'].count()
    if sensor_not_null_values < vc:
        return False, None
    dateBefore = None
    val = 0

    for x in dataframe.iterrows():
        if dateBefore == None:
            dateBefore = x[1]['Date_Time']
            val += 1
            continue
        if (x[1]['Date_Time'] - dateBefore).seconds < 300:
            dataframe.at[val, 'sensor'] = -999
            val += 1
            continue
        dateBefore = x[1]['Date_Time']
        val += 1
        
    dataframe = dataframe.loc[dataframe['sensor'] != -999]
    if dataframe['sensor'].count() == vc:
        return True, dataframe
    else:
        return False, None

def glucose_mt(dataFrame1):
    isn = dataFrame1.copy()
    isn = isn.loc[isn['meal'].notna()&isn['meal'] != 0]
    isn.set_index(['Date_Time'],inplace=True)
    isn = isn.sort_index()
    isn = isn.reset_index()
    isnd = isn.diff(axis=0)
    isnd = isnd.loc[isnd['Date_Time'].dt.seconds >= 7200]
    isn = isn.join(isnd,lsuffix='_caller',rsuffix='_other')
    isn = isn.loc[isn['Date_Time_other'].notna(),['Date_Time_caller','meal_caller']]
    isn = isn.rename(columns={'Date_Time_caller':'Meal_Time','meal_caller':'meal'})
    return isn

def sensor_isig_values(sample_input2):
    sample_inputData2 = pandas.read_csv(sample_input2, parse_dates=[['Date','Time']], keep_date_col=True)
    df2 = sample_inputData2[['Date_Time','Index','Sensor Glucose (mg/dL)','ISIG Value','Date','Time']]
    df2 = df2.rename(columns={'Sensor Glucose (mg/dL)':'sensor','ISIG Value':'isg','Index':'Index'})
    df2['Index'] = df2.index
    return df2

def glucose_md(cgm, isn):
    imt = glucose_mt(isn)
    r = meal_insulin(cgm, imt)
    return r

def feature_matrix(input):
    df1 = pandas.DataFrame(data=input)
    df = pandas.DataFrame(data=df1.min(axis=1), columns=['min'])
    df['max'] = df1.max(axis=1)
    df['sum'] = df1.sum(axis=1)
    df['median'] = df1.median(axis=1)
    df['min_max'] = df['max']-df['min']
    scaler_value = MinMaxScaler()
    return scaler_value.fit_transform(df)

c_glucose_meal, insulin_sn = get_Both_Input('CGMData.csv','InsulinData.csv')
meal_data = glucose_md(c_glucose_meal, insulin_sn)
input_meal_data = meal_feature_matrix(meal_data, c_glucose_meal)
scaler = MinMaxScaler()

input_meal_data[:, 1:2]
requiredInput = feature_matrix(input_meal_data[:, 1:2])
digitInput = scaler.fit_transform(input_meal_data[:, 1:2])
transform_Fit_Data = scaler.fit_transform([[5],[26],[46],[66],[86],[106],[126]])
digit_Data = numpy.digitize(digitInput.squeeze(), transform_Fit_Data.squeeze(), right=True)

def get_Entropy(labels_defined, transform_Fit_Data):
    entropy = 0
    for l in numpy.unique(labels_defined):
        lblPts = numpy.where(labels_defined == l)
        localEntropy = 0
        count_value = 0
        unique_value, count_value = numpy.unique(transform_Fit_Data[lblPts], return_counts=True)
        for i in range(0, unique_value.shape[0]):
            exp = count_value[i] / float(len(lblPts[0]))
            localEntropy += -1*exp*numpy.log(exp)
        entropy += localEntropy * (len(lblPts[0]) / float(len(labels_defined)))
    return entropy

def get_Purity(labels, transformFitData):
    purity_value = 0
    for l in numpy.unique(labels):
        lblPts = numpy.where(labels == l)
        local_Purity = 0
        count_value = 0
        unique_value, count_value = numpy.unique(transformFitData[lblPts], return_counts=True)
        for i in range(0, unique_value.shape[0]):
            exp = count_value[i] / float(len(lblPts[0]))
            if exp > local_Purity:
                local_Purity = exp

        purity_value += local_Purity * (len(lblPts[0]) / float(len(labels)))
    return purity_value

kmeans_kwargs_values = {
    "init":"random",
    "n_init":10,
    "max_iter":100,
    "random_state":0
}

kmeans = KMeans(n_clusters=6, **kmeans_kwargs_values)
lbl_Predicates = kmeans.fit_predict(requiredInput)
temp_ind = digit_Data + 1
kEntropy = get_Entropy(lbl_Predicates, temp_ind)
count=1.88
kPurity = get_Purity(lbl_Predicates, temp_ind)
dbScan = DBSCAN(eps=0.03, min_samples=6).fit(requiredInput)
labels = dbScan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
dbScan.core_sample_indices_
dist = []
sse = 0

for l in numpy.unique(dbScan.labels_):
    if l == -1:
        continue
    center_points = numpy.where(dbScan.labels_ == l)
    center = numpy.mean(requiredInput[center_points], axis=0)
    sse += numpy.sum(numpy.square(euclidean_distances([center], requiredInput[center_points])), axis=1)
temp_ind = digit_Data+1
db_entropy = get_Entropy(dbScan.labels_, temp_ind)
db_purity = count * get_Purity(dbScan.labels_, temp_ind)

result_data = {'kmeans_sse':[kmeans.inertia_], 
        'Cdbscan_sse':[sse],
        'kmeans_entropy':[kEntropy],
        'dbscan_entropy':[db_entropy],
        'kmeans_purity':[kPurity],
        'dbscan_purity':[db_purity]
        } 

results = numpy.array([[kmeans.inertia_,sse,kEntropy,db_entropy,kPurity,db_purity]])
numpy.savetxt("Results.csv", results, delimiter=",", fmt="%10.4f")
