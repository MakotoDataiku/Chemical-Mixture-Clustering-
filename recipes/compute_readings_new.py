# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
readings = dataiku.Dataset("readings")
df = readings.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
l = df.index.values
l = l * 0.01

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in range(0, len(l)):
    if i < 120:
        l[i] = l[i]
    elif i < 150:
        l[i] = l[i] * 1.2
    elif i < 180:
        l[i] = l[i] * 1.4
    elif i < 200:
        l[i] = l[i] * 1.6
    elif i < 220:
        l[i] = l[i] * 1.8
    elif i < 240:
        l[i] = l[i] * 2.0
    elif i < 300:
        l[i] = l[i] * 2.2
    elif i < 400:
        l[i] = l[i] * 2.3

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for c in df.columns.drop("Hz"):
    df[c] = df[c] * l

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
hz = df["Hz"] + 1200
df = df * 0.1 * 0.8
df["Hz"] = np.linspace(start=1200, stop=2400, num=401).astype(int)
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
readings_new = dataiku.Dataset("readings_new")
readings_new.write_with_schema(df)