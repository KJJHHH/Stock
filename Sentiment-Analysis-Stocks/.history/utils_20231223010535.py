import psutil
import pickle
import yfinance as yf
import pandas as pd
from googletrans import Translator


# check device usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

#
def get_ram_usage():
    return psutil.virtual_memory().percent

# translate
def trans(text_list):
    translator = Translator(service_urls=[
      'translate.google.com',
      'translate.google.co.kr',
    ])
    eng_texts = [translator.translate(i, dest='en').text for i in text_list]
    return eng_texts

#
def prc_data(stock):
    prc = yf.download(stock)
    prc["return"] = (prc["Adj Close"].shift(-1) - prc["Adj Close"])/prc["Adj Close"]
    prc = prc.reset_index().rename(columns={"Date":"time"})
    prc.time = prc.time.dt.date
    return prc

# read text data and transform time. here no missing data
def get_text_data(filename):
    with open(f'news/texts/{filename}.pk', 'rb') as f:
        data = pickle.load(f)
    data.drop("index", axis=1, inplace=True)
    data.time = data.time.apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M')).dt.date
    data.drop(["url"], axis=1, inplace=True)
    return data

#
def text_concatenate_diff_data(list_of_data):
    times = []
    for i in list_of_data:
        times += list(i.keys())
    times = set(times) # list(data_1.keys())
    data = dict()

    # combine data from different datas
    for time in times:
        content = ""
        for k in list_of_data:
            if time in k.keys():
                content += k[time]
        data[time] = [content]

    # data dictionary to dataframe
    data = pd.DataFrame.\
            from_dict(data, orient="index", columns=["content"]).\
            reset_index().\
            rename(columns={"index": "time"})
    return data

# make time in data cosistent of price datas' time
def process_time(prc, data):
    # update time in text data to match time in price data
    for index, (time, content) in data.iterrows():
        while time not in list(prc.time):
            time += datetime.timedelta(days=1)
        data.loc[index, 'time'] = time

    # check if all time in text data in price data
    for time in data.time:
        if time not in list(prc.time):
            print("error")
    
    return data
# combine the texts in the same date
def text_concatenate_within_same_date(data):
    concat_text = {}
    # concatenate news in the same date
    for index, (time, content) in data.iterrows():
        if index == 0:
            new_content = content
        else:
            if time == time_last:
                new_content += ("\n" + content)
            else:
                new_content = content
        
        concat_text[time] = new_content
        time_last = time 

    # dict to dataframe
    concat_text = pd.DataFrame(list(concat_text.items()), columns=["time", "content"])
    # check if time got duplicated
    print(f"duplicated time: {concat_text.time.duplicated().sum()}")
    return concat_text
# merge text and price data
def merge_prc_text(prc, data):
    valid_time = pd.merge(prc, data, on='time', how='left').\
                            fillna(method = "ffill").\
                            dropna()["time"]
    prc_text = pd.merge(prc, data, on='time', how='left')
    prc_text = prc_text[prc_text.time.isin(list(valid_time))]
    return prc_text