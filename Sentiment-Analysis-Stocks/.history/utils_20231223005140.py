# check device usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

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

def prc_data(stock):
    prc = yf.download(stock)
    prc["return"] = (prc["Adj Close"].shift(-1) - prc["Adj Close"])/prc["Adj Close"]
    prc = prc.reset_index().rename(columns={"Date":"time"})
    prc.time = prc.time.dt.date
    return prc

# read text data and transform timehere no missing data
def get_text_data(filename):
    with open(f'news/texts/{filename}.pk', 'rb') as f:
        data = pickle.load(f)
    data.drop("index", axis=1, inplace=True)
    data.time = data.time.apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M')).dt.date
    data.drop(["url"], axis=1, inplace=True)
    return data
