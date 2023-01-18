import pandas as ds
import datetime as dt
#Read Insulin Data from input CSV File
insulin_det = ds.read_csv("InsulinData.csv", low_memory=False)
insulin_det['Date']=ds.to_datetime(insulin_det['Date'], format="%m/%d/%Y")
insulin_det = insulin_det[["Date", "Time", "Alarm"]]
insulin_det['Time'] = ds.to_timedelta(insulin_det['Time'])

#Read CGM Data from input CSV File
cgm_det = ds.read_csv('CGMData.csv', low_memory=False)
cgm_det['Date']=ds.to_datetime(cgm_det['Date'], format="%m/%d/%Y")
cgm_det = cgm_det[["Date", "Time", "Sensor Glucose (mg/dL)","ISIG Value"]]
cgm_det.rename({'Sensor Glucose (mg/dL)': 'Glucose'}, axis=1, inplace=True)
cgm_det['Time'] = ds.to_timedelta(cgm_det['Time'])

#Update Data for auto and manual modes
UpdData = ds.DataFrame((insulin_det.loc[insulin_det['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']))
PLGM_Auto_Mode_OFF = UpdData.iloc[-1]["Date"]
PLGM_Auto_Mode_OFF_Time = UpdData.iloc[-1]["Time"]

manual_cgm = cgm_det[(cgm_det["Date"] < PLGM_Auto_Mode_OFF) | ((cgm_det["Date"] == PLGM_Auto_Mode_OFF) & (cgm_det["Time"] < PLGM_Auto_Mode_OFF_Time))]
auto_cgm = cgm_det[(cgm_det["Date"] > PLGM_Auto_Mode_OFF) | ((cgm_det["Date"] == PLGM_Auto_Mode_OFF) & (cgm_det["Time"] >= PLGM_Auto_Mode_OFF_Time))]

#Main logic to extract metrics based on time intervals and other parameters
list_manual_mode = []
list_auto_mode = []
q = 0
e = 0
manual_date_data = manual_cgm["Date"].unique()
auto_date_data = auto_cgm["Date"].unique()
time_intervals = ['over_night','day_time','whole_day']
levels_of_glucose = ['hyperglycemia','hyperglycemia critical','range','range secondary','hypoglycemia level 1','hypoglycemia level 2']
op_mode = ['manual','auto']

while q<len(op_mode):
    if op_mode[q] == 'auto':
        end_list = auto_cgm
        list_of_dates = auto_date_data
    else:
        end_list = manual_cgm
        list_of_dates = manual_date_data
    i = 0
    while i<len(time_intervals):
        if time_intervals[i] == 'over_night':
            Time_Interval_CGM = end_list[(end_list['Time']>='0:00:00') & (end_list['Time']<= '6:00:00')]
        if time_intervals[i] == 'day_time':
            Time_Interval_CGM = end_list[(end_list['Time']>'6:00:00') & (end_list['Time']<= '23:59:59')]
        if time_intervals[i] == 'whole_day':
            Time_Interval_CGM = end_list
        i+=1
        r = 0
        while r<len(levels_of_glucose):
            e = 0
            if levels_of_glucose[r] == 'hyperglycemia':
                for k in list_of_dates:
                    time_based_result = Time_Interval_CGM[Time_Interval_CGM["Date"] == k]
                    e = e + (time_based_result[time_based_result["Glucose"] > 180].shape[0]/float(288))*100
                e = e/(float(list_of_dates.shape[0])*1.0)
                if op_mode[q] == 'manual':
                    list_manual_mode.append(e)
                elif op_mode[q] == 'auto':
                    list_auto_mode.append(e)
            elif levels_of_glucose[r] == 'hyperglycemia critical':
                for k in list_of_dates:
                    time_based_result = Time_Interval_CGM[Time_Interval_CGM["Date"] == k]
                    e = e + (time_based_result[time_based_result["Glucose"] > 250].shape[0]/float(288))*100
                e = e/(float(list_of_dates.shape[0])*1.0)
                if op_mode[q] == 'manual':
                    list_manual_mode.append(e)
                else:
                    list_auto_mode.append(e)
            elif levels_of_glucose[r] == 'range secondary':
                for k in list_of_dates:
                    time_based_result = Time_Interval_CGM[Time_Interval_CGM["Date"] == k]
                    e = e+(time_based_result[(time_based_result["Glucose"] >= 70) & (time_based_result["Glucose"] <= 150)].shape[0]/float(288))*100
                e = e/(float(list_of_dates.shape[0])*1.0)
                if op_mode[q] == 'manual':
                    list_manual_mode.append(e)
                elif op_mode[q] == 'auto':
                    list_auto_mode.append(e)
            elif levels_of_glucose[r] == 'hypoglycemia level 2':                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                for k in list_of_dates:
                    time_based_result = Time_Interval_CGM[Time_Interval_CGM["Date"] == k]
                    e = e+(time_based_result[time_based_result["Glucose"] < 54].shape[0]/float(288))*100
                e = e/(float(list_of_dates.shape[0])*1.0)
                if op_mode[q] == 'manual':
                    list_manual_mode.append(e)
                elif op_mode[q] == 'auto':
                    list_auto_mode.append(e)
            elif levels_of_glucose[r] == 'range':
                for k in list_of_dates:
                    time_based_result = Time_Interval_CGM[Time_Interval_CGM["Date"] == k]
                    e = e+(time_based_result[(time_based_result["Glucose"] >= 70) & (time_based_result["Glucose"] <= 180)].shape[0]/float(288))*100
                e = e/(float(list_of_dates.shape[0])*1.0)
                if op_mode[q] == 'manual':
                    list_manual_mode.append(e)
                elif op_mode[q] == 'auto':
                    list_auto_mode.append(e)
            elif levels_of_glucose[r] == 'hypoglycemia level 1':
                for k in list_of_dates:
                    time_based_result = Time_Interval_CGM[Time_Interval_CGM["Date"] == k]
                    e = e+(time_based_result[time_based_result["Glucose"] < 70].shape[0]/float(288))*100
                e = e/(float(list_of_dates.shape[0])*1.0)
                if op_mode[q] == 'manual':
                    list_manual_mode.append(e)
                elif op_mode[q] == 'auto':
                    list_auto_mode.append(e)
            r+=1               
    q+=1
#Append and save end results to a CSV File

end_outcome = ds.DataFrame()
end_outcome['Manual'] = list_manual_mode
end_outcome['Auto'] = list_auto_mode
end_outcome = end_outcome.transpose()
end_outcome.to_csv("Results.csv",index=False,header=False)