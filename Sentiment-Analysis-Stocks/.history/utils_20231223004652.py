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