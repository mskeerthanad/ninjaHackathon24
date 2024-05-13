import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Example input data
input_data = {
    'Date': ['2023-01-06'],  # Example date for prediction (not used in prediction)
    'Commodity': ['Onion'],  # Example value for Commodity
    'State': ['Karnataka'],  # Example value for State
    'District': ['Bangalore'],  # Example value for District
    'Market': [0],
    'Population': [13606753],
    'MinPrice':[500],  # Example value for MinPrice
    'MaxPrice': [800],  # Example value for MaxPrice
    'TotalArrival': [2983]  # Example value for TotalArrival
}

# Create a DataFrame from input_data
input_df = pd.DataFrame(input_data)

# Assuming these are the features your model expects
features = ['Commodity', 'State', 'District', 'Market', 'Population', 'MinPrice', 'MaxPrice', 'TotalArrival']

# Transform categorical columns using LabelEncoder
label_encoders = {}
for feature in features:
    le = LabelEncoder()
    input_df[feature] = le.fit_transform(input_df[feature])
    label_encoders[feature] = le

# Convert input_df to numpy array and reshape
input_array = input_df[features].to_numpy().reshape(-1, 1, len(features), 1)
file_name = "/Users/nc23629-keerthana/Desktop/hack/flask-server/models/your_model_filename.pkl"
model = open(file_name, 'wb')
# Now you can use input_array to predict with your Keras model
predicted_price = model.predict(input_array)
print(predicted_price)