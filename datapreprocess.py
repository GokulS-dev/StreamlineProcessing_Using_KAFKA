import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('datasets\computed_insight_success_of_active_sellers.csv')

# Handle missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Correct errors or inconsistencies
# No specific correction needed based on the provided column names

# Scale numerical features
scaler = MinMaxScaler()
df[['totalunitssold', 'meanproductprices', 'meanretailprices', 'averagediscount', 'meandiscount', 'meanproductratingscount', 'totalurgencycount', 'urgencytextrate']] = scaler.fit_transform(df[['totalunitssold', 'meanproductprices', 'meanretailprices', 'averagediscount', 'meandiscount', 'meanproductratingscount', 'totalurgencycount', 'urgencytextrate']])

# Save preprocessed data to a new CSV file
df.to_csv('preprocessed_summer_clothing_sales.csv', index=False)
