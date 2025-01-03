import pandas as pd
from confluent_kafka import Consumer, KafkaError
import json
from sklearn.preprocessing import StandardScaler
import pickle

# Kafka broker address
bootstrap_servers = 'localhost:9092'

# Kafka topic to consume from
topic = 'tests'

# Consumer group ID
group_id = 'my_consumer_group'

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create Consumer instance
consumer = Consumer({
    'bootstrap.servers': bootstrap_servers,
    'group.id': group_id,
    'auto.offset.reset': 'earliest'  # Start reading at the beginning of the topic
})

# Subscribe to the Kafka topic
consumer.subscribe([topic])

# Function to validate incoming message
def validate_message(msg):
    try:
        # Parse JSON message
        data = json.loads(msg.value().decode('utf-8'))
        
        # Perform validation checks
        # Example: Ensure required fields are present and have correct data types
        if 'orderid' not in data or 'listedproducts' not in data:
            print('Error: Missing required fields')
            return False
        # Add more validation checks as needed
        
        return True
    except Exception as e:
        print('Error:', e)
        return False

# Function to cleanse incoming message
def cleanse_message(msg):
    try:
        # Parse JSON message
        data = json.loads(msg.value().decode('utf-8'))
        
        # No need for data cleansing in this example
        
        return data
    except Exception as e:
        print('Error:', e)
        return None

# Function for feature extraction and preprocessing
# Function for feature extraction and preprocessing
def preprocess_data(data):
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data, index=[0])
        
        # Drop irrelevant fields (e.g., orderid and totalunitssold) before prediction
        df.drop(columns=['orderid', 'totalunitssold'], inplace=True)
        
        return df
    except Exception as e:
        print('Error during data preprocessing:', e)
        return None

    
# Function for model prediction
def predict_with_model(data):
    try:
        # Predict using the loaded model
        prediction = model.predict(data)
        return prediction
    except Exception as e:
        print('Error during prediction:', e)
        return None

# Poll for new messages, validate, cleanse them, preprocess, and process them
try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition
                continue
            else:
                print(msg.error())
                break
        
        # Print out the message content for inspection
        print("Received message:", msg.value().decode('utf-8'))
        
        # Validate message
        if validate_message(msg):
            # Cleanse message (if needed)
            cleansed_data = cleanse_message(msg)
            if cleansed_data:
                # Preprocess data
                preprocessed_data = preprocess_data(cleansed_data)
                if preprocessed_data is not None:
                    # Predict using the model
                    prediction = predict_with_model(preprocessed_data)
                    # Display preprocessed data and prediction
                    print('Preprocessed data:', preprocessed_data)
                    print('Prediction:', prediction)
except KeyboardInterrupt:
    pass
finally:
    # Close the consumer to gracefully shut down
    consumer.close()
