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

I created the '03_car_emission_LR.py' script based on my learnings from the Machine Learning with Python course, part of the IBM data science centification in Coursera.
The dataset I used in this script is from: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv
The dataset includes the CO2 emissions of cars along with car information such as engine size, cylinders, fuel consumption in city and highway.
80% of the data is randomly selected as the training set; and the rest as the test dataset. Model evaluation was performed using the test dataset.

I tested 3 multivariate linear regression models to estimate the car CO2 emissions. 
In Model 1: I used engine size and cylinders to estimate emissions;
In Model 2: I used engine size, cylinders as well as the fuel consumption in city and highway to estimate emissions;
In Model 3: I used same features as in Model 2 but this time used a 2nd degree polynomial fitting on engine size.
To get a sense of accuracy of the models, I created scatter plots of predicted and actual emissions vs engine size on the test dataset.
I also looked at mean absolute and squared errors, and R2 score to compare the model results quantitatively.
While addition of fuel consumption improved the predictions largely, the polynomial fitting of the engine size did not result in significant improvements.

------------------------------------------------------------------------------------
