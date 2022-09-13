import streamlit as st
import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

# header = st.container()
# dataset = st.container()
# features = st.container()
# model_training = st.container()pip install -U scikit-learn

#Here you can add css to beautify your dashboard.
# st.markdown(
#     """
#     <style>
#     .main{
#         backgorund-color : #F5F5F5:
#     }
#     </style>
#     """
#     unsafe_allow_html=True
# )

@st.cache()
def get_data(filename):
    taxi_data = pd.read_csv(filename)

    return taxi_data

with st.container():
    st.title('Welcome to my first streamlit dashboard')
    st.write("I am trying to build my first streamlit dashboard")
 

with st.container():
    st.header('NYC Taxi data set')
    st.write('I found this data on kaggle')
    #call data through function
    taxi_data = get_data('data/taxi_data.csv')

    # taxi_data = pd.read_csv('data/taxi_data.csv')
    st.write(taxi_data.head())
    st.subheader('Trip Duration of NYC Taxi Trips')
    trip_duration_graph= pd.DataFrame(taxi_data['trip_duration'].value_counts()).head(50)
    st.bar_chart(trip_duration_graph)

with st.container():
    st.write("Feature Extraction")
    st.markdown('* **Feature extraction stage 1:** Here are the extracted features')
    st.markdown('* **Feature extraction stage 2:** Here are the extracted features')

with st.container():
    st.write("Lets add a machine learning model")
    #lets add columns. Here we are making 2 columns. First column is sel_col and the other is disp_col

    sel_col , disp_col = st.columns(2)
    #Add content in first column

    max_depth = sel_col.slider('What should be the max depth of the model?', min_value = 10 , max_value = 100, value= 20, step = 10)
    n_estimator = sel_col.selectbox('How many trees should there be?', options = [100,200,300,'No limit'])

    sel_col.text("Here is the list of features in our data")
    sel_col.write(taxi_data.columns)


    input_feature = sel_col.text_input('Which feature be used as an input','trip_duration')
    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)
    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_duration']]

    regr.fit(X,y.values.ravel())
    prediction = regr.predict(y)
    # Add things in 2nd columns
    disp_col.subheader('MAE of the model is: ')
    disp_col.write(mean_absolute_error(y, prediction))
    disp_col.subheader('MSE of the model is: ')
    disp_col.write(mean_squared_error(y, prediction))
    disp_col.subheader('R2 of the model is: ')
    disp_col.write(r2_score(y, prediction))

    #For deployment make requrement.txt file by instaling pip install pipreqs in the same folder where your app is
    #type: python -m  pipreqs.pipreqs . 