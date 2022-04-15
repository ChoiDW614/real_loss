# -*- coding: euc-kr -*-

#import csv
#line_list = []
#with open("C:/Users/djw04/Desktop/PythonQuant/PER_ROA.csv") as csv_file:
#    csv_reader = csv.reader(csv_file, delimiter=',')
#    for row in csv_reader:
#        if '' in row:
#            pass
#        else:
#            line_list.append(row)

#    df = pd.DataFrame(data=line_list[1:], columns=line_list[0])
#    print(df.head())

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/djw04/Desktop/PythonQuant/PER_ROA.csv", engine="python")
print(df.head())



