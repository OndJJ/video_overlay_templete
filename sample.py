import cv2
import numpy as np
import os
import pandas as pd

# txt file load

# data = pd.read_csv('파일경로', sep = "\t", , engine='python', encoding = "인코딩방식")
data = pd.read_csv(r'01_202109081705528_2\01_202109081705528_2_det.txt',
                   sep=",", engine='python', encoding="cp949")
print(data.shape)

print(data.head(30))
print(data.info())
