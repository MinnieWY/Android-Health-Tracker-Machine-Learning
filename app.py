from flask import Flask, jsonify, request
import requests
from datetime import datetime, timedelta
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report

app = Flask(__name__)

def retrive_sleep_data(headers,date):
    url = f"https://api.fitbit.com/1.2/user/-/sleep/date/{date}.json"
    response = requests.get(url, headers=headers)
    data = response.json()
    sleep_log_df = pd.DataFrame(data['sleep'])

    sleep_log_df = sleep_log_df[sleep_log_df['infoCode'] == 0]
    sleep_log_df = sleep_log_df[sleep_log_df['isMainSleep'] == True]
    sleep_log_df.drop('logId', axis=1, inplace=True)
    sleep_log_df.drop('logType', axis=1, inplace=True)
    sleep_log_df.drop('infoCode', axis=1, inplace=True)
    sleep_log_df.drop('type', axis=1, inplace=True)
    sleep_log_df.drop('isMainSleep', axis=1, inplace=True)
    sleep_log_df['duration'] = sleep_log_df['duration'] / (1000 * 60)
    sleep_log_df = sleep_log_df.rename(columns={'timeInBed': 'minutesInBed'})
    sleep_log_df.drop('startTime', axis=1, inplace=True)
    sleep_log_df.drop('endTime', axis=1, inplace=True)
    sleep_log_df.drop('efficiency', axis=1, inplace=True)
    sleep_log_df.drop('minutesAfterWakeup', axis=1, inplace=True)
    sleep_log_df.drop('minutesAsleep', axis=1, inplace=True)
    sleep_log_df.drop('minutesAwake', axis=1, inplace=True)
    sleep_log_df.drop('minutesToFallAsleep', axis=1, inplace=True)
    sleep_log_df.drop('minutesInBed', axis=1, inplace=True)
    
    return sleep_log_df

def get_sleep_stage_count(raw_df):
    
    sleep_log_df = pd.DataFrame(raw_df['levels'])
    new_df = raw_df.loc[:, ['levels', 'dateOfSleep']]

    summary_data_list = []
    dates = []  # Store unique dateOfSleep values
    new = pd.DataFrame()
    for index, row in new_df.iterrows():
        string_df = row['levels']
        summary_data = string_df['data']
        df_summary = pd.DataFrame(summary_data)
        df_summary['dateOfSleep'] = row['dateOfSleep']
        new = pd.concat([new, df_summary], axis=0)

    new_df = pd.pivot_table(new, index='dateOfSleep', columns='level', aggfunc='size', fill_value=0)
    new_df.reset_index(drop=True, inplace=True)
    
    new_df = new_df.rename(columns={'wake': 'wake_count'})
    new_df = new_df.rename(columns={'light': 'light_count'})
    new_df = new_df.rename(columns={'deep': 'deep_count'})
    new_df = new_df.rename(columns={'rem': 'rem_count'})
    
    return new_df

def get_sleep_summary(raw_df):
    sleep_log_level = pd.DataFrame(raw_df['levels'])
    new_df = raw_df.loc[:, ['levels', 'dateOfSleep']]

    summary_data_list = []
    dates = []  # Store unique dateOfSleep values

    for index, row in new_df.iterrows():
        string_df = row['levels']
        summary_data = string_df['summary']
        df_summary = pd.DataFrame(summary_data)
        summary_data_list.append(df_summary)
        dates.append(row['dateOfSleep'])

    summary_data_modified = []

    for df_summary, date in zip(summary_data_list, dates):
        summary_data_flattened = {}
        for element, attributes in df_summary.items():
            for attribute, value in attributes.items():
                if attribute != 'thirtyDayAvgMinutes':
                    new_attribute = element + '_' + attribute
                    summary_data_flattened[new_attribute] = value

        # Create a new DataFrame with the modified data
        df_summary_modified = pd.DataFrame(summary_data_flattened, index=[0])
        df_summary_modified['dateOfSleep'] = date

        summary_data_modified.append(df_summary_modified)

    summary_df = pd.concat(summary_data_modified, ignore_index=True)
    summary_df.drop('deep_count', axis=1, inplace=True)
    summary_df.drop('light_count', axis=1, inplace=True)
    summary_df.drop('rem_count', axis=1, inplace=True)
    summary_df.drop('wake_count', axis=1, inplace=True)
    summary_df.drop('dateOfSleep', axis=1, inplace=True)
    
    return summary_df

def retrive_breathing_data(headers,date):
    breathing_rate_data = []

    url = f"https://api.fitbit.com/1/user/-/br/date/{date}/all.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    breathing_rate_data.extend(data['br'])

    breathing_rate_df = pd.DataFrame(breathing_rate_data)

    expanded_data = breathing_rate_df['value'].apply(pd.Series)

    breathing_data = pd.concat([breathing_rate_df['dateTime'], expanded_data], axis=1)

    breathing_data['deepSleepSummary'] = breathing_data['deepSleepSummary'].apply(lambda x: x['breathingRate'] if isinstance(x, dict) else x)
    breathing_data['remSleepSummary'] = breathing_data['remSleepSummary'].apply(lambda x: x['breathingRate'] if isinstance(x, dict) else x)
    breathing_data['lightSleepSummary'] = breathing_data['deepSleepSummary'].apply(lambda x: x['breathingRate'] if isinstance(x, dict) else x)

    breathing_data = breathing_data.rename(columns={'deepSleepSummary': 'deep_breath'})
    breathing_data = breathing_data.rename(columns={'remSleepSummary': 'rem_breath'})
    breathing_data = breathing_data.rename(columns={'lightSleepSummary': 'light_breath'})
    
    breathing_data.drop('fullSleepSummary', axis=1, inplace=True)
    
    return breathing_data

def retrieve_hrv(headers,date):
    url = f"https://api.fitbit.com/1/user/-/hrv/date/{date}/all.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    dataset = data['hrv'][0]['minutes']
    hrv_df = pd.DataFrame(dataset)

    hrv_df = pd.concat([hrv_df.drop(['value'], axis=1), hrv_df['value'].apply(pd.Series)], axis=1)
    hrv_df.drop('coverage', axis=1, inplace=True)
    hrv_df['date'] = pd.to_datetime(hrv_df['minute'])
    hrv_df = hrv_df.groupby(hrv_df['date'].dt.date).mean()
    hrv_df = hrv_df.reset_index()
    hrv_df.drop('date', axis=1, inplace=True)

    return hrv_df

def retrieve_resting_hr(headers, date):
    url = f"https://api.fitbit.com/1.2/user/-/activities/heart/date/{date}/1m.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    activites = pd.DataFrame(data['activities-heart'])

    new = pd.DataFrame()
    for index, row in activites.iterrows():
        value_data = row['value']
        if 'restingHeartRate' in value_data:
            summary_data = value_data['restingHeartRate']
        else:
            summary_data = np.nan
        summary_df = pd.DataFrame({'restingHeartRate': [summary_data]})
        summary_df['dateTime'] = row['dateTime']
        new = pd.concat([new, summary_df], axis=0)
    
    return new

def retrieve_spo2(headers, date):
    url = f"https://api.fitbit.com/1/user/-/spo2/date/{date}/all.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    dataset = data['minutes']
    spo2_df = pd.DataFrame(dataset)

    data = spo2_df['value'].mean()
    spo2_df = pd.DataFrame({'spo2': [data]})

    return spo2_df

def retrieve_steps(headers, date):
    url = f"https://api.fitbit.com/1/user/-/activities/date/{date}.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    steps = data['summary']['steps']
    steps_df = pd.DataFrame({'steps':steps}, index=[0])

    return steps_df

def retrieve_data(access_token):
    today = datetime.today()
    yesterday= today - timedelta(days=1)
    
    date_str_today = today.strftime('%Y-%m-%d')
    date_str_yesterday = yesterday.strftime('%Y-%m-%d')

    headers = {"Authorization": f"Bearer {access_token}"}
    
    sleep_df = retrive_sleep_data(headers,date_str_today)
    sleep_count = get_sleep_stage_count(sleep_df)
    sleep_summary = get_sleep_summary(sleep_df)
    breathing_df = retrive_breathing_data(headers,date_str_today)
    hrv_df = retrieve_hrv(headers,date_str_today)
    hr_df = retrieve_resting_hr(headers,date_str_today)
    spo2_df = retrieve_spo2(headers,date_str_today)
    steps_df = retrieve_steps(headers,date_str_yesterday)
    sleep_df.drop('levels', axis=1, inplace=True)

    df = pd.concat([sleep_df, breathing_df, hrv_df, hr_df,spo2_df,steps_df,sleep_count,sleep_summary], axis=1)
    df.drop('dateTime', axis=1, inplace=True)
    df.drop('dateOfSleep', axis=1, inplace=True)
    
    df['rem_proportion'] = df['rem_minutes'] / df['duration']
    df['deep_proportion'] = df['deep_minutes'] / df['duration']
    df['light_proportion'] = df['light_minutes'] / df['duration']
    df['wake_proportion'] = df['wake_minutes'] / df['duration']
    
    df = df[['rmssd', 'hf', 'lf','spo2','steps','rem_breath','rem_minutes','rem_count','rem_proportion','deep_breath', 'deep_minutes', 'deep_count', 'deep_proportion','light_breath','light_minutes','light_count','light_proportion','wake_minutes','wake_count','wake_proportion','restingHeartRate','duration']]
    return df


@app.route("/predict", methods=["POST"])
def predict_stress():
    access_token = request.form.get('accessToken')

    if not access_token:
        return jsonify({'error': 'Access token is required'}), 404

    features = retrieve_data(access_token)
    if features is None:
        return jsonify({'error': 'Features is null'}), 404
    
    try:
        rf_model = load('./notebooks/model.pkl')
        predictions = rf_model.predict(features)

        result = str(predictions[0])
        return result

    except Exception as e:
        error_message = f"An error occurred during the prediction: {str(e)}"
        return jsonify({'error': error_message}), 500

if __name__ == "__main__":
    app.run()