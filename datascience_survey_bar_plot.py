import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library

df_survey = pd.read_csv('https://cocl.us/datascience_survey_data', index_col=0)
print('Data downloaded and read into a dataframe!')

# sort dataframe in descending order of 'Very interested'
df_survey.sort_values(by='Very interested', ascending=False, inplace=True)

# convert numbers into percentages of total number of respondents
total_respond = 2233
df_survey_percentage = df_survey.div(total_respond).mul(100).round(decimals=2)

# create bar plot and set title and legend
ax = df_survey_percentage.plot(kind='bar', 
                               figsize=(20,8), 
                               width=0.8, 
                               color=['#5cb85c','#5bc0de','#d9534f'],
                               fontsize=14)                              
ax.set_title("Percentage of Respondents' Interest in Data Science Areas", fontsize=16)
ax.legend(labels=df_survey_percentage.columns, fontsize=14)

# display percentages above the bars.
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height}', (x + width/2, y + height + 1), ha='center', fontsize=14)
    
# remove left, top, and right borders.
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_visible(False)
