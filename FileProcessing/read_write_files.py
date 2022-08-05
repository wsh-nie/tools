import os
import json
import csv
import pandas as pd
import pandas._libs.lib as lib
import random


def write_json(data_dict, file_path):
    json_str = json.dumps(data_dict)
    with open(file_path, 'w') as json_file:
        json_file.write(json_str)
    json_file.close()


def read_json(file_path):
    with open(file_path, 'r') as json_File:
        data_dict = json.load(json_File)
    json_File.close()
    return data_dict


def read_csv(file_path, sep=lib.no_default, header="infer", index_col=None, usecols=None):
    pd_data = pd.read_csv(file_path, sep=sep, header=header, index_col=index_col, usecols=usecols)
    pd_data_dict = pd_data.to_dict('split')
    return pd_data_dict['data']


def write_csv(data_array, file_path, header=None):
    """
    :param data_array: structured ndarray, sequence of tuples or dicts, or DataFrame
    :param file_path: saving file path
    :param header: sequence, default None
    :return: No return
    """
    with open(file_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        if header is not None:
            writer.writerow(header)
        for row_data in data_array:
            writer.writerow(row_data)


def read_txt(file_path):
    txt_data = []
    for line in open(file_path, 'r'):
        txt_data.append(line[:-1]) # windows操作系统下去除'\n'

    return txt_data


def write_txt(txt_data, file_path)
    with open(file_path, 'w') as fw:
        fw.writelines(txt_data)
    fw.close()


if __name__ == '__main__':
    data = []
    header = ['col_1', 'col_2', 'col_3']
    for i in range(100):
        data.append([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])
    print(data)
    write_csv(data, './test.csv', header=header)
    read_data = read_csv('./test.csv')
    print(read_data)
