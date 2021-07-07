import os 
import pathlib
import boto3 
from PIL import Image, ImageDraw, ImageFilter
import pathlib
import argparse
from functions import *
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from statdepth import FunctionalDepth

# All the imaging data will be here
here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', 'data')

parser = argparse.ArgumentParser(description='Generate growth and irregularity curves for the given experiment UUID on the braingeneers/imaging/ s3 bucket.')
parser.add_argument('-ID','--uuid', help='Experiment UUID', required=True)
parser.add_argument('-T', '--times', help='Path to list of times that the organoids are observed at. Can be found in S3. MUST be formatted like the file example_times.txt', required=True)
parser.add_argument('-CC', '--cameras', help='Path to list of cameras that the organoids are observed at. Can be found in S3. MUST be formatted like the file example_cameras.txt', required=True)
parser.add_argument('-CR', '--credentials', help='Path to AWS credentials for braingeneers')
args = vars(parser.parse_args())

N = 2
UUID = str(args['uuid'])
TIMES = str(args['times'])
CAMS = str(args['cameras'])
CREDS = str(args['credentials'])
BUCKET = 'braingeneersdev'
WINDOW_LEN = 9
POLY_ORDER = 3

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', 'data')

with open(CAMS) as f:
    cameras = [c.strip() for c in f]

with open(TIMES) as f:
    times = [t.strip() for t in f]

with open(CREDS) as f:
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
        Key=os.path.join('jlehrer', new_name)
    )

def download_camera(camera, time):
    '''Download all 10 images from the given camera at the given time'''

    print(f'Downloading data from camera {camera[-2:]} at time {time}')
    os.makedirs(os.path.join(here, '..', 'data', camera, time), exist_ok=True)
    
    for t in range(1, 11):
        if not os.path.isfile(os.path.join(here, '..', 'data', camera, time, f'{t}.jpg')):
            try:
                s3.Bucket('braingeneers').download_file(
                    Key=os.path.join('imaging', UUID, 'images', time, camera, f'{t}.jpg'), 
                    Filename=os.path.join(here, '..', 'data', camera, time, f'{t}.jpg')
                )
            except:
                print(f'Cannot download {t}.jpg')
                continue

# Download subset of data, generate composites, and remove the originals
for cam in cameras:
    for time in times:
        if not os.path.isfile(os.path.join(data_path, cam, time, 'composite.jpg')):
            path = os.path.join(data_path, cam, time)
            download_camera(cam, time)

            # Make sure the folder isn't empty -- some timepoints are missing all the images
            if len(os.listdir(path)) != 0:
                generate_composite(path)
                remove_originals(path)

print('Done with downloading script...')
print('Starting plot generation')

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

