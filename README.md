# Milestone 4 

<details open>
<summary><h2>Introduction</h2></summary>

TYPE THE INTRODUCTION HERE

</details>

<details open>
<summary><h2>Methods</h2></summary>

## Data Exploration 





&nbsp;
## Pre-Processing 

## **Correlation Matrix**






&nbsp;
## Model 1: Neural Network 






&nbsp;
## Model 2: K Nearest Neighbor 

The second model we decided to use is the KNN, K Nearest Neighbor. This model was selected for predicting weather-related variables due to its simplicity and effectiveness in capturing patterns in data without assuming an underlying distribution. The KNN algorithm works by finding the closest training examples in the feature space and predicting the target value based on the average of these neighbors. This approach makes it particularly well-suited for problems where the relationship between features and the target variable is non-linear.

&nbsp;

We split the data into training and test sets with an 80-20 split. This step was essential for evaluating the model's performance on unseen data. We then trained the KNN model on the training data using five neighbors.
&nbsp;
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

```
&nbsp;
We made predictions on the test set and evaluated the model using Mean Squared Error (MSE), a common metric for regression models.
&nbsp;

```
y_pred = knn_model.predict(X_test)

mse_temp = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0])
mse_wind = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1])
mse_sea = mean_squared_error(y_test.iloc[:, 2], y_pred[:, 2])

print(f"Mean Squared Error for Temperature: {mse_temp}")
print(f"Mean Squared Error for Wind Speed: {mse_wind}")
print(f"Mean Squared Error for Sea Pressure: {mse_sea}")
```
&nbsp;
We visualized the actual versus predicted values to gain further insights into the model's performance
&nbsp;
```

plt.figure(figsize=(15, 5))

sorted_indices = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[sorted_indices]

#Temperature
plt.subplot(1, 3, 1)
plt.plot(X_test_sorted, y_test.iloc[sorted_indices, 0], label='Actual Temperature', color='blue')
plt.plot(X_test_sorted, y_pred[sorted_indices, 0], label='Predicted Temperature', color='red', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Temperature')
plt.title('Actual vs Predicted Temperature')
plt.legend()

#Wind Speed
plt.subplot(1, 3, 2)
plt.plot(X_test_sorted, y_test.iloc[sorted_indices, 1], label='Actual Wind Speed', color='blue')
plt.plot(X_test_sorted, y_pred[sorted_indices, 1], label='Predicted Wind Speed', color='red', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Wind Speed')
plt.title('Actual vs Predicted Wind Speed')
plt.legend()

#Sea Pressure
plt.subplot(1, 3, 3)
plt.plot(X_test_sorted, y_test.iloc[sorted_indices, 2], label='Actual Sea Pressure', color='blue')
plt.plot(X_test_sorted, y_pred[sorted_indices, 2], label='Predicted Sea Pressure', color='red', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Sea Pressure')
plt.title('Actual vs Predicted Sea Pressure')
plt.legend()

plt.tight_layout()
plt.show()
```



</details>

<details open>
<summary><h2>Results</h2></summary>

## Model 1 


## Model 2: KNN Final Model 
&nbsp;
Detailed Analysis
The Mean Squared Error (MSE) values for the predictions were:
&nbsp;

- Temperature: 0.005237
  &nbsp;
- Wind Speed: 0.018628
 &nbsp;
- Sea Pressure: 0.006608

&nbsp;
Temperature Prediction:
&nbsp;

The low MSE of 0.005237 for temperature indicates that the KNN model's predictions were highly accurate. The visualization showed a close alignment between the actual and predicted temperature values, suggesting that the model effectively captured the temporal patterns and variations in temperature. This performance demonstrates the model's ability to handle the non-linear relationships present in the temperature data.
&nbsp;

Wind Speed Prediction:
&nbsp;

The MSE of 0.018628 for wind speed was higher than that for temperature and sea pressure. The visualization revealed that while the model could capture the general trend of wind speed, there were some noticeable deviations. These deviations indicate that the model struggled to predict wind speed as accurately as it did for temperature and sea pressure. Wind speed can be more variable and influenced by a wider range of factors, which may not have been fully captured by the features used in the model.
&nbsp;

Sea Pressure Prediction:
&nbsp;

The MSE of 0.006608 for sea pressure was also low, indicating good model performance. The close alignment between the actual and predicted values in the visualization suggests that the model was effective in capturing the trends in sea pressure data. The performance for sea pressure was similar to that for temperature, indicating that the model was consistent in predicting variables that may have more stable patterns compared to wind speed.
&nbsp;

Overall Performance:

&nbsp;
The KNN model showed varying performance across different weather-related variables. It performed best for **temperature prediction**, followed by sea pressure, and then wind speed. The low MSE values for temperature and sea pressure indicate that the model was able to accurately capture the patterns in these variables. However, the higher MSE for wind speed suggests that there is room for improvement in capturing the more complex patterns associated with wind speed.








</details>

<details open>
<summary><h2>Discussion</h2></summary>

</details>

<details open>
<summary><h2>Conclusion</h2></summary>

</details>

<details open>
<summary><h2>Statement of Collaboration</h2></summary>

</details>










# Milestone 3
The code starts by preparing the data for analysis. It uses a specific DataFrame, hourly_avg_temp, which includes the average hourly temperature. The 'hour' column is separated from the features for later use. The features are then scaled to a range between 0 and 1 using the MinMaxScaler from sklearn, resulting in a normalized dataset which can help compare the two different information with each other and how it relates to time. The processed features and the 'hour' column are then concatenated to form the final DataFrame, final_df.

Next, the code sets up the neural network for training. It imports necessary libraries such as pandas, numpy, and various modules from keras for building the neural network. The prepared data (processed_df) is used as the input features (X), while the original features are used as the target labels (y). The dataset is split into training and test sets, with 10% of the data reserved for testing. A sequential neural network model is built with one input layer, two hidden layers, each containing 12 neurons and using the sigmoid activation function. The output layer also uses the sigmoid activation function. The model is compiled using the SGD optimizer with a learning rate of 0.3 and the categorical crossentropy loss function, and then trained for 100 epochs with a batch size of 32 and a validation split of 10%.

After training, the model's performance is evaluated on the test set. Predictions are made using the trained model, and the true and predicted class labels are determined. A confusion matrix is computed to assess the performance of the model. This matrix is visualized using ConfusionMatrixDisplay from sklearn, displaying the results in a heatmap format. The confusion matrix helps in understanding the accuracy and error rates of the model's predictions.
In summary, the code involves a comprehensive process of data preparation, neural network training, and performance evaluation. The data is first merged and normalized, and then a neural network is built and trained to predict weather-related features. The model's accuracy is assessed using a confusion matrix, providing insights into the model's predictive capabilities and highlighting areas for potential improvement. This process demonstrates the application of machine learning techniques to analyze and predict complex weather patterns.

With our results, we noticed that our model has around an accuracy of 5.8912^-6 which means our model is underfitting. This means that we have to improve our model to have a higher accuracy to have our model to be appropriate-fitting. Another improvement we can make is to do hyperparameter tuning to determine which parameters are the best suited for our model.

## Milestone 2
We preprocessed our data to answer the question: since 2000, has the Earth gotten hotter? We'll do this by comparing every hourly collectively since 2000, to see if they steadily increase. We started by collecting a dataset called 'weather' with 43 features and over 261,000 rows. Given the abundance of data, we had to process it to our liking. First, we made a list of features we wanted to drop since they do not correlate with answering our question. These features were: ['Unnamed: 32', 'Unnamed: 41', 'metar', 'metar_origin', 'pressure_change_code', 'weather_cond_code', 'pressure_change_code', 'visibility', 'cloud_layer_1', 'cloud_layer_2', 'cloud_layer_3', 'wind_cardinal_direction', 'precip_accum_one_hour', 'cloud_layer_1_code', 'cloud_layer_2_code', 'cloud_layer_3_code', 'heat_index']. After removing the unnecessary features, we addressed the issue of missing data by dropping NA values from our main column, 'air_temp', and then filling in every numeric column with NA values using the corresponding column mean. We converted 'air_temp' to a numeric column to ensure it was in the correct format.

Next, we converted and changed the format of the 'date_time' column in 'weather.csv' to 'pd.to_datetime', allowing us to utilize the date more precisely. Using the 'date_time' column, we created four additional columns: 'year', 'month', 'day', and 'hour'. With these new columns, we calculated each hourly average temperature by grouping by 'year' and 'month', ‘day’, and ‘hour’ and calculating the mean.

In our first graph. The code uses Python's `matplotlib` and `seaborn` to create a line graph showing how three-hour accumulated precipitation changes over the years. First, it sets up the plot and sorts the data by year. It then plots the years on the x-axis and the precipitation values on the y-axis using a green line. The graph shows a sharp drop in precipitation around 2000, followed by ups and downs, and a slight increase in recent years. This helps visualize the trend of three-hour precipitation over time.

In our second graph. We also used Python's `matplotlib` and `seaborn` to create a line graph showing how air temperature changes over the years. First, it sets up the plot and sorts the data by year. It then plots the years on the x-axis and the air temperature values on the y-axis using a red line. The graph shows fluctuations in air temperature from 1998 to 2025, with notable peaks and troughs. This helps visualize the trend of air temperature over time.

For our third graph. The code uses `matplotlib` to create a line graph showing monthly average temperatures over time. It sets up a large figure, then plots the year and month on the x-axis and the average temperature on the y-axis using blue circles and lines. The graph shows regular seasonal cycles with temperatures rising and falling each year. There are clear peaks and troughs, indicating the typical seasonal changes. The labels and title help explain the graph, making it easy to see how average temperatures have varied over the years.

For our fourth graph we did a scatter plot. This code uses `matplotlib` to create a scatter plot showing yearly average temperatures over time. It sets up a figure, then plots the years on the x-axis and the average temperatures on the y-axis using blue dots. The plot is labeled to indicate that the data points represent actual temperature data. The x-axis is labeled 'Year' and the y-axis is labeled 'Yearly Average Temperature.' The graph shows individual data points for each year's average temperature, revealing trends and variations over time. The grid and legend make it easier to read and understand the plot. This visual representation helps in identifying patterns in yearly average temperatures from 2000 to 2025.

For our fifth graph we did a pairplot. The code merges weather data with monthly average temperatures, converts the 'date_time' column to extract year, month, day, and hour, and converts specified columns to numeric types. It then drops rows with missing values in these columns. Finally, it creates a pair plot using `seaborn` to show scatter plots and histograms of the relationships between different weather variables. The pair plot helps visualize how these variables, such as temperature, humidity, and wind speed, relate to each other.

For our last graph we did a correlation matrix. The code calculates the correlation matrix for various weather variables and visualizes it using a heatmap. The correlation matrix shows how strongly each pair of variables, such as temperature, humidity, and wind speed, are related. The heatmap uses colors to represent the correlation values, with annotations displaying the exact values. This helps quickly identify which variables have strong positive or negative correlations.
