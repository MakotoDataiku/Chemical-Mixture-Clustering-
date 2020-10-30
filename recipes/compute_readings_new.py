# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
readings = dataiku.Dataset("readings")
readings_df = readings.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

readings_new_df = readings_df # For this sample code, simply copy input to output


# Write recipe outputs
readings_new = dataiku.Dataset("readings_new")
readings_new.write_with_schema(readings_new_df)
