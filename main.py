import streamlit as st
from ml_util import *

st.header("Inspecting the data")

data = pd.read_csv('data.csv')

data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y %H:%M")
data.set_index('Date', inplace=True)

st.write(data)


# print(data)

# Check for missing values
st.write(f"missing values: {data.isna().sum()}")
st.write("no missing values")
# print(data.describe())

st.subheader("Data Distribution of Features")

plot_data_distributions(data)

st.write("Features 10-13 are already scaled. The rest will need to be scaled. The data is normally distributed in the other features. we can use a minmax scaler to get it on the same scale as features 10-13")


st.subheader("Feature Correlation Matrix")

plot_correlation_matrix(data)

st.write("We can see that some features are highly correlated with the target, and some are not. We also see that some features are highly correlated with each other")

st.write("Features, 1,4, and 8 are highly correlated with each other, and they all are highly correlated with the target")
st.write("Features 2 and 5, and features 3 and 6 are highly correlated with each other as well. We may be able to remove or combine some of these redundant features to improve our model, but it is difficult to know what we can remove or manipulate without context of the data")


st.subheader("Check for outliers with boxplots")

plot_feature_boxplots(data)

st.write("We can see that there are outliers on a few of the features, but no major outliers. features 1,4,8 especially have some outliers. It is hard to tell without context whether these extreme values are important for predicting the target ")
st.write("for now we can go ahead without removing them, and come back to this later.")

st.subheader("Pre-processing the data")

st.write("Now that we've had a good look at the data, we can preprocess it and then train a base model")
st.write("As mentioned, we will scale features 1-9")


features_to_drop = ['Feature 7', 'Feature 10', 'Feature 2']
# features_to_drop=[]
features_to_scale = ['Feature 1', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 8', 'Feature 9']


X_preprocessed, y = preprocess_data(data, features_to_drop, features_to_scale)


st.subheader("Train model")

st.write("we are dealing with 15-minute time-series data. in order to predict the target at some time, we likely will have to look back a certain number of timesteps")
st.write("We can use a LSTM model in order to train the model on this temporal information")
st.write("however, without context of the data, we cannot know exactly what time window to use for training. We can test different time windows")


st.write("After training the model several times with different hyperparameters, I found the following to be the most optimal:")

params = {
    "time_steps": 12,
    "epochs": 300,
    "learning rate": 0.0005,
    "LSTM layers": [100, 50, 50, 1],
    "featuers dropped": [2, 7, 10]
}

st.table(params)

# time_steps, epochs, learning rate 
time_steps=12
epochs=300
learning_rate=0.0005

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_preprocessed, y, time_steps)


st.write("Training...")
history, model = train_model(X_train, X_val, y_train, y_val, epochs, learning_rate, time_steps)

val_loss = np.min(history.history['val_loss'])
st.write(val_loss)

plot_training_validation_curve(history)

st.write("The training and validation loss are improving steadily overtime, with no signs of overfitting")

st.write("All of the metrics are good. The MSE is not much higher than the MAE, which means that the errors in the predictions are not very large")
st.write("An R2 score of 0.945 is quite high, indicating that the model explains a large portion of the variance in the data")

st.write("let's use the model generate predictions on the test data, and some analysis on the results")


run_tests(model, y_test, X_test)


st.write("The residuals plot seems to be quite uniform. This means that the model captures the variance in the data quite well.")

st.write("The autocorrelation plot, and the durban-watson statistic indicate that there is no autocorrelation in the residuals")
st.write("This means that the residuals are mostly noise, and are not correlated to any time-frames in the time series")

st.subheader("improvements")

st.write("Overall, the results are solid, but the model can be improved, and the results can be further analyzed")
st.write("In terms of the results, it would be beneficial to further analyze them to detect time-specific patterns. It's possible that my errors have a pattern of occuring at certain times of the day, month, or year.")
st.write("This analysis would help guide improvements of the model. For example, an ensemble method could be used, where one model is trained on specific time periods, or time windows, and its results are combined with those of another model")

st.write("My current results show that the model was learning well, and was continuing to improve at 300 epochs. With more time, I would have liked to experiment with more complex model architectures and longer training periods. It would be interesting to also test different models, such as a transformer.")

st.write("I could have also performed further feature engineering: Adding features from rolling statistics in the data may help the model learn if there are patterns in the data. Additional combinations or removal of features may help learning as well.")



st.header("Additional Question: 3D data")

st.write("If I was given similar data, but for 100 points on a 10x10 grid, I would take a similar approach but with a convolutional neural net. This would help not only capture the temporal relationships in the data, but it would learn the spatial relationships as well")
st.write("In order to achieve this, I would use tensorflow's ConvLSTM. This maintains the LSTM's core functionality and purpose, but uses convolutional layers in its gates, allowing the model to capture spatial data")
st.write("The preprocessing of the data would be very similar, but with 3D data instead of 2D. This would result in data with the shape: (batch_size, timesteps, 10, 10), which can be fed into the LSTM")
st.write("The implementation of the model would be as follows:")


code='''model=Sequential([
    #create a convolutional layer for each input (time_steps, 10x10 grid)
    #adjust the filter size, kernel size, and normalization / pooling depending on the nature of the data
    TimeDistributed(Conv2D(filter, kernel, activation), input_shape=(12,10,10)) 
    TimeDistributed(BatchNormalization()) 
    TimeDistributed(MaxPooling2D(pool_size))

    # Now, the LSTM layers will capture relationships in the data across time / space
    ConvLSTM2D(filters, kernel)
    ConvLSTM2D(filters, kernel)
    # add more layers if needed

    # output a 10x10 grid of predictions, or whatever shape you want the prediction to be
    Dense(10,10)

])'''
st.code(code, language='python')