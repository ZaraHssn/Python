# This script creates a dataframe with total number of crimes in each of the 10 neighborhoods in San Francisco and creates a Choropleth map map to visualize the data.
# The San Francisco crime dataset is loaded from: https://cocl.us/sanfran_crime_dataset
# and the GeoJSON file (defining boundaries of the neighborhoods in San Francisco) is taken from: https://cocl.us/sanfran_geojson

import pandas as pd # data structure library

# create a dataframe from San Francisco crime dataset that lists total number of crimes in each neighborhood

# read the crime incindent dataset
df_incidents = pd.read_csv('https://cocl.us/sanfran_crime_dataset')

# count the number of incidents in each neighborhood and write to a dataframe
df_incidents_counts = pd.DataFrame(df_incidents.PdDistrict.value_counts())

# reset the index range to go from 0 to number of neighborhoods
df_incidents_counts.reset_index(inplace = True)

# change column names
df_incidents_counts.rename(columns={'index':'Neighborhood','PdDistrict':'Count'}, inplace=True)

import folium

# download geojson file
!wget --quiet https://cocl.us/sanfran_geojson -O sanfran_geo.json

sanfran_geo = r'sanfran_geo.json'

# San Francisco latitude and longitude values
Sanfran_lat = 37.77
Sanfran_lon = -122.42

# let Folium determine the scale
world_map = folium.Map(location=[Sanfran_lat, Sanfran_lon], zoom_start=12)
world_map.choropleth(
    geo_data=sanfran_geo,
    data=df_incidents_counts,
    columns = ['Neighborhood','Count'],
    key_on='feature.properties.DISTRICT',
    fill_color = 'YlOrRd', 
    fill_opacity = 0.7, 
    line_opacity = 0.2,
    legend_name='Crime Rate in San Francisco',
)
world_map
