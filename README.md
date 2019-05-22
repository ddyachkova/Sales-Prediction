This test problem is related to a time series prediction for items that are going to be purchased in the upcoming 4 weeks. 
I began by analyzing the historical data. 
First, one can notice that there are 3 items that the data describe - Brezel, Buttercroissant and Apfeltasche (see the histogram of the unique values). I transformed the original dataset so that it is more similar to the form in which the predicted data is presented - time series is given as days rather than time of each day for each of the purchases, and the numbers of purchases are counted for each of the items.
I then analyzed the trends of the data to check whether:
- there is seasonality, 
- the data is stationary 
- the target variables are autocorrelated.  

As the plots suggest, autocorrelation is weak, there is no trend and the dataset is stationary. The latter is supported by the result of the Augmented Dickey-Fuller (ADF) test.
Then I converted the time series data that we are given to treat the problem as a supervised learning one - so that there are labels for the model to learn. It is done by using the shift() function that creates copies of columns that are pushed forward or pulled back. 
After that, I normalized that data so that in each column the values are scaled between 0 and 1, even though the ranges of the columns are comparable, with normalization the results were more accurate. 

For this problem, I used Long short-term memory network (LSTM), which is a type of RNN. 
The input to such network must be in the following shape: the batch size, the number of time-steps and the number of units in one input sequence. I ended up having 50 neurons in each of the layers, having added two LSTM layers and the Dense layer at the end. As it was mentioned in the description of the task, we use RMSE as the metric, so the loss function is MSE. I also added a Dropout layer to ensure that there is no overfitting (the comparison of RMSE for training and test datasets shows that), and a Flatten layer. The latter was added, as the output of LSTM had to be flattened out before the Dense layer. 
RMSE was around 4 and 5, which is not sufficient for Buttercroissant and Brezel data, as the mean of those columns is 91.03 and 95.4 respectively. However, for Apfeltasche, where the mean of the column is 15.04 it is a lot. This can be explained by insufficient amount of data - this item was the least populat among the purchases as the very first histogram shows. 
The way that the model works is the following. It is trained on multiple variables - all the data that is given but uses one variable at a time as the label. Therefore, I had to fit the model three times to predict each of the labels to then inverse the scaling all together and add the prediction as a batch for each of the days. I did it for each of the days and added the values to the prediction dataset. 
