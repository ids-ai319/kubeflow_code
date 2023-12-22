import csv
from pathlib import Path
# 讀入檔案
rows =[]
# 最初讀入時，header也要讀入
skip_num = 0
for file in Path("data").glob("*.csv"):
    f = open(file)
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num <= skip_num:
            continue
        rows.append(row)
    f.close()
    skip_num = 1
# 寫入檔案
f = open("./data/predict_sum.csv",mode='a', newline="")
writer = csv.writer(f)
for row in rows:
    writer.writerow(row)
f.close()