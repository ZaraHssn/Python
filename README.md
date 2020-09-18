# Python
Python Scripts for Data Science and Machine Learning

------------------------------------------------------------------------------------

The '01_datascience_survey_bar_plot.py' script reads a survey of 2,233 respondents on 6 data science topics:

1. Big Data (Spark / Hadoop)
2. Data Analysis / Statistics
3. Data Journalism
4. Data Visualization
5. Deep Learning
6. Machine Learning

The respondents were given three options for each topic: Very Interested, Somewhat interested, and Not interested.

The script:
1. loads the survey results saved to a csv file from: https://cocl.us/datascience_survey_data;
2. converts the numbers to percentages of the total respondents;
3. summarizes the survey results in a bar plot.

I made this script as part of 'Data Visualization with Python' certification with IBM.

------------------------------------------------------------------------------------

The '02_crime_map_sf.py' script creates a dataframe with total number of crimes in each of the 10 neighborhoods in San Francisco and plots a Choropleth map to visualize the data.
The San Francisco crime dataset is loaded from: https://cocl.us/sanfran_crime_dataset
and the GeoJSON file (defining boundaries of the neighborhoods in San Francisco) is taken from: https://cocl.us/sanfran_geojson

I made this script as part of 'Data Visualization with Python' certification with IBM.

------------------------------------------------------------------------------------

I created the script '03_car_emission_LR.py' based on my learnings from Machine Learning with Python (IBM).
The dataset I used in this script is from: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv
The dataset includes the CO2 emissions of cars along with car information such as engine size, cylinders, fuel consumption in city and highway.

Two multivariate linear regression models are implemented to estimate the car CO2 emissions. 
In the first model two variables are used: engine size and cylinders, while in the second model fuel consumption in city and highway are also included.
The improvements achieved by addition of the fuel consumption parameters are examined by comparing the mean square errors as well as 
scatter plots of the predicted and actual emission rates for the test data set. 

------------------------------------------------------------------------------------
