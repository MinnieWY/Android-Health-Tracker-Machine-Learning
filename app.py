from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import pandas as pd

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
    sleep_log_df.drop('levels', axis=1, inplace=True)
    sleep_log_df['startTime'] = pd.to_datetime(sleep_log_df['startTime'])
    sleep_log_df['endTime'] = pd.to_datetime(sleep_log_df['endTime'])
    sleep_log_df['duration'] = sleep_log_df['duration'] / (1000 * 60)
    sleep_log_df = sleep_log_df.rename(columns={'timeInBed': 'minutesInBed'})
    sleep_log_df.drop('startTime', axis=1, inplace=True)
    sleep_log_df.drop('endTime', axis=1, inplace=True)
    sleep_log_df.drop('dateOfSleep', axis=1, inplace=True)
    return sleep_log_df

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
    breathing_data['fullSleepSummary'] = breathing_data['deepSleepSummary'].apply(lambda x: x['breathingRate'] if isinstance(x, dict) else x)
    breathing_data['lightSleepSummary'] = breathing_data['deepSleepSummary'].apply(lambda x: x['breathingRate'] if isinstance(x, dict) else x)

    breathing_data = breathing_data.rename(columns={'deepSleepSummary': 'deepSleep'})
    breathing_data = breathing_data.rename(columns={'remSleepSummary': 'remSleep'})
    breathing_data = breathing_data.rename(columns={'fullSleepSummary': 'fullSleep'})
    breathing_data = breathing_data.rename(columns={'lightSleepSummary': 'lightSleep'})

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
    breathing_df = retrive_breathing_data(headers,date_str_today)
    hrv_df = retrieve_hrv(headers,date_str_today)
    spo2_df = retrieve_spo2(headers,date_str_today)
    steps_df = retrieve_steps(headers,date_str_yesterday)

    df = pd.concat([sleep_df, breathing_df, hrv_df,spo2_df,steps_df], axis=1)
    df.drop('dateTime', axis=1, inplace=True)
    return df


@app.route('/predict')
def predict_stress():
    access_token = request.form.get('accessToken')
    features = retrieve_data(access_token)

    return features.to_json()

if __name__ == "__main__":
    app.run()