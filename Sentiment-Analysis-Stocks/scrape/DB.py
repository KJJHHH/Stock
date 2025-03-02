import json
import pandas as pd
import numpy as np
import datetime
import sqlalchemy
import pickle
from sqlalchemy import create_engine, VARCHAR

def conn(database_name):
    host = 'localhost'
    port = "3306"
    user = 'root'
    pwd = 'test123'
    database_name = database_name
    con = create_engine('mysql+pymysql://'+user+':'+pwd+'@'+host+':'+port+'/'+database_name)
    return con

    
def get_item(filename):
    # file_path = 'C:/Users/USER/Desktop/sentiment_Text/news/texts/2023121521_anue_友達.json'
    json_file = open(f"texts/{filename}", 'r', encoding="utf-8", errors="replace")
    dict_posts = json.load(json_file)
    json_file.close()
    return dict_posts

def to_database(conn, search, filename):
    """
    search = "顯示器"
    filename = '2023121713_anue_顯示器.json'
    """
    dict_posts = get_item(filename)

    urls = pd.DataFrame(dict_posts.keys(), columns = ["url"])
    times_contents = pd.DataFrame(dict_posts.values(), columns=["time", "content"])
    data = pd.merge(urls, times_contents, left_index=True, right_index=True)

    max_date = datetime.datetime.strptime(max(data['time']), "%Y/%m/%d %H:%M").strftime("%Y%m%d")
    min_date = datetime.datetime.strptime(min(data['time']), "%Y/%m/%d %H:%M").strftime("%Y%m%d")
    f"anue_{search}_{max_date}to{min_date}"
    table_name = f"anue_{search}_{str(max_date)}to{str(min_date)}"
    data.to_sql(table_name, con=conn, if_exists='replace')

def load_database(con, filename):
    sql_query = f'SELECT * FROM {filename}'
    data = pd.read_sql(sql_query, con)
    return data