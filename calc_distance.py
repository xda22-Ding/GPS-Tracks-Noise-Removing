import sys
import os
import gpxpy
import numpy as np 
import pandas as pd
import xml.dom.minidom as md
from math import cos, asin, sqrt, pi
from pykalman import KalmanFilter


def distance(points):
### Inpired by https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
### Thanks to Professor's hint https://coursys.sfu.ca/2022fa-cmpt-353-d1/forum/77#post-78
    lat_dist = pd.DataFrame({'x': points['lat']}).astype(float)
    lat_dist['x1'] = lat_dist['x'].shift(-1)
    lat_dist = lat_dist.dropna().copy()
    lat_dist['distance'] = (lat_dist['x1'] - lat_dist['x']).abs()


    lon_dist = pd.DataFrame({'y': points['lon']}).astype(float)
    lon_dist['y1'] = lon_dist['y'].shift(-1)
    lon_dist = lon_dist.dropna().copy()
    lon_dist['distance'] = (lon_dist['y1'] - lon_dist['y']).abs()
    
    p = pi/180
    df1 = pd.DataFrame({"a":lat_dist['distance']*p,"b":lat_dist['x']*p,"c":lat_dist['x1']*p,"d":(lon_dist['y1']-lon_dist['y'])*p})
    df1 = df1.apply(np.cos,axis=1)
    
    df1['result'] =  0.5 - df1['a']/2 + df1['b'] * df1['c'] * (1-df1['d'])/2
    df1['inter_result'] = df1['result'].apply(np.sqrt,axis = 1)
    df1['final_distance'] = df1['inter_result'].apply(np.arcsin,axis = 1)
    return df1['final_distance'].sum()*12742000
    

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.7f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def smooth(points):
    # Kalman Filter
    points['lat'] = points['lat'].astype(float)
    points['lon'] = points['lon'].astype(float)


    kalman_data = points[['lat', 'lon', 'Bx', 'By']]


    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([0.2, 0.2,0.2, 0.2]) ** 2 
    transition_covariance = np.diag([0.2, 0.2, 0.2, 0.2]) ** 2 
    transition = [[1,0,6*10**(-7),29*10**(-7)], [0,1,-43*10**(-7),12*10**(-7)], [0,0,1,0], [0,0,0,1]] 

    kf = KalmanFilter(

        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition

    )

    kalman_smoothed, _ = kf.smooth(kalman_data)
    return kalman_smoothed





def main():
    input_gpx = sys.argv[1]
    input_csv = sys.argv[2]

    points = md.parse(input_gpx)  # parse an XML file by name
    coordinates = points.getElementsByTagName('trkpt')
    time_data = points.getElementsByTagName('time')
    lat = []
    lon = []
    time = []

    for i in range(coordinates.length):
        lat.append(coordinates[i].getAttribute('lat'))
        lon.append(coordinates[i].getAttribute('lon'))
        time.append(time_data[i].firstChild.data)


    points = pd.DataFrame({'datetime':time,'lat': lat,'lon':lon})
    points['datetime'] = pd.to_datetime(points['datetime'], utc=True)


    points = points.set_index('datetime')



    sensor_data = pd.read_csv(input_csv, parse_dates=['datetime']).set_index('datetime')
    points['Bx'] = sensor_data['Bx']
    points['By'] = sensor_data['By']
    
    dist = distance(points)    
    print(f'Unfiltered distance: {dist:.2f}')

    
    smoothed_points = smooth(points)
    smoothed_points = pd.DataFrame(smoothed_points, columns = ['lat','lon','Bx','By'])  
    smoothed_dist = distance(smoothed_points)
    print(f'Filtered distance: {smoothed_dist:.2f}')

    output_gpx(smoothed_points, 'out.gpx')
    

if __name__ == '__main__':
    main()
