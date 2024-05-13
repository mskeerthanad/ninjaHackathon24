
import pandas as pd


merged_df = pd.read_csv('/Users/nc23629-keerthana/Desktop/hack/flask-server/Data-processed/Agmarknet - Sheet1 (1).csv')

merged_df['Total_Arrival'] = merged_df['Total_Arrival'].str.replace(',','').astype(int)

merged_df['Total_Arrival'] = merged_df['Total_Arrival']/10

merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%d-%b-%Y')



# Assuming merged_df is your DataFrame containing the dataset

# Convert Date column to datetime format using the correct format
merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%d-%b-%Y')

def get_average_total_arrival(date_str, state, district):
    input_date = pd.to_datetime(date_str, format='%d-%b-%Y')
    
    # Filter the dataset based on input date, state, and district
    filtered_df = merged_df[(merged_df['Date'].dt.month == input_date.month) & 
                             (merged_df['Date'].dt.year == input_date.year) &
                             (merged_df['State'] == state) &
                             (merged_df['District'] == district)]
    
    if filtered_df.empty:
        # If no exact match for input date, go to the last available date in the dataset
        last_date_row = merged_df[merged_df['Date'] == merged_df['Date'].max()]
        filtered_df = merged_df[(merged_df['Date'].dt.month == last_date_row['Date'].dt.month.values[0]) &
                                 (merged_df['Date'].dt.year == last_date_row['Date'].dt.year.values[0]) &
                                 (merged_df['State'] == state) &
                                 (merged_df['District'] == district)]
    
    if not filtered_df.empty:
        # Calculate average of Total_Arrival for the entire month
        avg_total_arrival = filtered_df['Total_Arrival'].mean()
        print("avg_total_arrival",avg_total_arrival)
        return avg_total_arrival
    else:
        return None  # No data found for the input criteria





def get_modal_price(date_str, state, district, commodity):
    print("Hello Everyone")
    input_date = pd.to_datetime(date_str, format='%d-%b-%Y')
    print(input_date)
    # Filter the dataset based on input date, state, district, and commodity
    filtered_df = merged_df[(merged_df['Date'].dt.month == input_date.month) &
                             (merged_df['Date'].dt.year == input_date.year) &
                             (merged_df['State'] == state) &
                             (merged_df['District'] == district) &
                             (merged_df['Commodity'] == commodity)]
    print(filtered_df,"filtered_df")

    if filtered_df.empty:
        print("empty")
        # If no exact match for input date, go to the last available date in the dataset
        last_date_row = merged_df[merged_df['Date'] == merged_df['Date'].max()]
        filtered_df = merged_df[(merged_df['Date'].dt.month == last_date_row['Date'].dt.month.values[0]) &
                                 (merged_df['Date'].dt.year == last_date_row['Date'].dt.year.values[0]) &
                                 (merged_df['State'] == state) &
                                 (merged_df['District'] == district) &
                                 (merged_df['Commodity'] == commodity)]

    if not filtered_df.empty:
        # Get the ModalPrice for the filtered row
        print(filtered_df)
        modal_price = filtered_df['ModalPrice'].values[0]
        print(modal_price,"modal_price")
        return modal_price
    else:
        return None  # No data found for the input criteria


merged_df['Population'] = merged_df['Population'].str.replace('.', '').astype(int)


# for col in ['MinPrice', 'MaxPrice', 'ModalPrice']:
#     # Calculate average of top and bottom values if present
#     top_value = merged_df[col].dropna().iloc[-1] if not pd.isna(merged_df[col].iloc[-1]) else None
#     bottom_value = merged_df[col].dropna().iloc[0] if not pd.isna(merged_df[col].iloc[0]) else None
#     if top_value is not None and bottom_value is not None:
#         average_value = (top_value + bottom_value) / 2
#     else:
#         average_value = top_value or bottom_value

#     # Fill NaN values with the calculated average or a single value
#     merged_df[col].fillna(average_value, inplace=True)

# merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.strftime('%Y-%m-%d')
# merged_df['Commodity'] = merged_df['Commodity'].astype('category').cat.codes
# merged_df['State'] = merged_df['State'].astype('category').cat.codes
# merged_df['District'] = merged_df['District'].astype('category').cat.codes
# merged_df['Market'] = merged_df['Market'].astype('category').cat.codes