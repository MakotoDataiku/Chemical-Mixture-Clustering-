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
for c in df.columns.drop("Hz"):
    df[c] = df[c] * l

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
readings_new = dataiku.Dataset("readings_new")
readings_new.write_with_schema(df)