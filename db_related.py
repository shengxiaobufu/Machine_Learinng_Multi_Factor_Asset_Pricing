import pymysql
import pandas as pd


def get_connect(host="localhost", user='root', psd='Yangfan1108',
                db='master_paper', charset='utf8'):
    connection = pymysql.connect(host=host,
                                 user=user,
                                 password=psd,
                                 db=db,
                                 charset=charset,
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection


def insert_many(sql, data_list):
    conn = get_connect()
    with conn.cursor() as cursor:
        cursor.executemany(sql, data_list)
        print('sql committing...')
        conn.commit()
        print('commited, affected rows: ', conn.affected_rows())

    conn.close()


def select_data(sql):
    conn = get_connect()
    with conn.cursor() as cursor:
        cursor.execute(sql)
        print('sql committing...')
        conn.commit()
        data = pd.DataFrame(cursor.fetchall())
    conn.close()
    return data
