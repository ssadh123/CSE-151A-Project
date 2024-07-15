We preprocessed our data by merging the two datasets, weather.csv and wildfire.csv, to provide us the opportunity to find correlations between the high temperatures and wind speed in Los Angeles and the number of acres that the wildfire spread throughout the area.

During the exploration of the weather dataset, we’ve come across many unnecessary columns that didn’t serve any purpose in answering our hypothesis, so we decided to limit our feature to only five key features ‘Data_time’, ‘air_temp’, ‘wind_speed’, ‘relative_humidity’, and ‘sea_level_pressure’. On the other hand our wildfire dataset we decided to get five features regarding 'incident_name','incident_date_created','incident_date_extinguished', 'incident_county', and 'incident_acres_burned'. On top of these features we filtered out any incidents that didn’t happen in Los Angeles County.

To merge the two datasets, we standardized their date formats. We decided to merge the datasets by matching the date of each recording. However, the formats were different between the two datasets, so we had to change them to a consistent format of month, day, and year. We also removed the time component to ensure that rows would match correctly.

First, we converted and changed the format of the `date_time` column in `weather.csv`. Similarly, we changed the format of the `incident_date_created` column in `wildfire.csv`. After that, we created new columns in both datasets called `date`, which contained only the date without the time component.


We then merged the two datasets, resulting in a new dataframe called merge_df that included information from both `weather.csv` and `wildfire.csv`. Additionally, we changed the format of the `incident_date_extinguished` column to match the new format of `incident_date_created`. Finally, we dropped the original `date_time` column and set the index to the `date` column.

We used ‘pd.to_datetime’ to convert the data columns of ‘weather_df’ and ‘wildfire_LA’ into datetime objects, then merged the two dataframes on the ‘data’ column. Next we defined a subset of columns that are of interest for correlation analysis, and dropped rows with missing values in these columns and filtered out non-numeric entries. For visualization, we generated a pairplot using seaborn to explore the relationship between our selected variables. Additionally, we generated a heatmap of the correlation matrix to illustrate the linear relationships between the variables.

Based on the heatmap generated, we can see some correlations between wind_speed & air_temp and acres_burned and wind_speed.
