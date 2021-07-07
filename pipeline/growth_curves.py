import pandas as pd
import os
import numpy as np
import pathlib
import cv2 as cv
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go
import plotly.express as px 
from statdepth import FunctionalDepth
import boto3
from .functions import *

N = 3
BUCKET = 'braingeneersdev'
BUCKET_SUBPATH = 'jlehrer/organoids'
WINDOW_LEN = 9
POLY_ORDER = 3

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', 'data')

with open(os.path.join(here, '..', 'times.txt')) as f:
    times = [t.strip() for t in f]
    
with open(os.path.join(here, '..', 'cameras.txt')) as f:
    cameras = [c.strip() for c in f]

with open(os.path.join(here, '..', 'credentials')) as f:
    aws_access_key_id, aws_secret_access_key = [line.strip() for line in f]

s3 = boto3.resource(
    's3',
    endpoint_url="https://s3.nautilus.optiputer.net",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

def upload(f, new_name):
    s3.Bucket(BUCKET).upload_file(
        Filename=f,
        Key=os.path.join(BUCKET_SUBPATH, new_name)
    )

# Compute growth curves
df = pd.DataFrame()

for cam in cameras:
    print(f'Calculating growth curve for {cam}')
    try:
        df[cam] = growth_curve(cam, times, skip=1)
    except Exception as e:
        print(str(e))
        continue

irreg_df = pd.DataFrame()

for cam in cameras:
    print(f'Calculating irregularity curve for {cam}')
    try:
        irreg_df[cam] = irregularity_curve(cam, times, skip=1)
    except Exception as e:
        print(e)
        continue

d = FunctionalDepth([df], relax=True, quiet=False)
d_irreg = FunctionalDepth([irreg_df], relax=True, quiet=False)

fig_df = d.plot_deepest(
    N, 
    title=f'{N} deepest growth curves', 
    xaxis_title='Time', 
    yaxis_title='Relative area',
    return_plot=True
)

fig_irreg_df = d_irreg.plot_deepest(
    N, 
    title=f'{N} deepest irregularity curves', 
    xaxis_title='Time', 
    yaxis_title='Irregularity',
    return_plot=True
)

fig_df.write_image(f'{N}_deepest_growth.png', scale=3)
fig_irreg_df.write_image(f'{N}_deepest_growth_irreg.png', scale=3)

print('Uploading growth curves')
upload(
    f'{N}_deepest_growth.png',
    f'{N}_deepest_growth.png',
)

upload(
    f'{N}_deepest_growth_irreg.png',
    f'{N}_deepest_growth_irreg.png',
)

# Now apply savgol filter to data and generate the depth plots again
d._orig_data = savgol_filter(df, WINDOW_LEN, POLY_ORDER)
d_irreg._orig_data = savgol_filter(irreg_df, WINDOW_LEN, POLY_ORDER)

fig_df = d.plot_deepest(
    N, 
    title=f'{N} deepest growth curves (Savgol filter, order {POLY_ORDER}', 
    xaxis_title='Time', 
    yaxis_title='Relative area',
    return_plot=True
)

fig_irreg_df = d_irreg.plot_deepest(
    N, 
    title=f'{N} deepest irregularity curves (Savgol filter, order {POLY_ORDER}', 
    xaxis_title='Time', 
    yaxis_title='Irregularity',
    return_plot=True
)

fig_df.write_image(f'{N}_deepest_growth_savgol.png', scale=3)
fig_irreg_df.write_image(f'{N}_deepest_growth_irreg_savgol.png', scale=3)

print('Uploading irregularity curves')
upload(
    f'{N}_deepest_growth_savgol.png',
    f'{N}_deepest_growth_savgol.png',
)

upload(
    f'{N}_deepest_growth_irreg_savgol.png',
    f'{N}_deepest_growth_irreg_savgol.png',
)
