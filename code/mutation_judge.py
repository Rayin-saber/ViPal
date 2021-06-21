# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:10:43 2020

@author: Rui Yin and Luo Zihan
"""

import csv
import os
import pandas as pd

os.chdir("E:/2019 NTU/Project/Program/Prior Knowledge/CTD/data/")


mutation_site = {
    "PB2": [357, 591, 627, 701],
    "PA": [35, 97, 224, 383],
    "HA": [138, 143, 163, 185, 187, 190, 222, 225, 226, 228],
    "NA": [223, 275],
    "NS1": [92]
}

mutation_amio = {
    "PB2": ['N', 'K', 'K', 'N'],
    "PA": ['L', 'I', 'P', 'P'],
    "HA": ['A', 'G', 'E', 'S', 'D', 'D', 'G', 'D', 'L', 'S'],
    "NA": ['I', 'Y'],
    "NS1": ['E']
}

row_num = 0
csvFile = open("flu_feature.csv", "r")
reader = csv.reader(csvFile)
with open("flu_feature_mutation.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for row in reader:
        for segment in mutation_site.keys():
            segment_file = pd.read_csv('csv/' + str(segment) + '_data_with_site.csv')
            for num in range(len(mutation_site[segment])):
                # print(site)
                res = segment_file.loc[row_num, [str(segment) + ':' + str(mutation_site[segment][num])]]
                if res[0] == mutation_amio[segment][num]:
                    row.append(1)
                else:
                    row.append(0)
        writer.writerow(row)
        row_num += 1
