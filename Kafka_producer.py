from confluent_kafka import Producer
import pandas as pd

# Load the dataset (replace 'dataset.csv' with the actual path to your dataset)
dataset_path = 'preprocessed_summer_clothing_sales.csv'
df = pd.read_csv(dataset_path)

# Kafka broker address
bootstrap_servers = 'localhost:9092'

# Kafka topic to publish to
topic = 'tests'

# Create Kafka Producer instance
producer = Producer({'bootstrap.servers': bootstrap_servers})

# Function to publish data to Kafka topic
def publish_to_kafka(row):
    try:
        # Convert row to JSON format and send to Kafka topic
        producer.produce(topic, value=row.to_json().encode('utf-8'))
        print("Published message to Kafka topic:", row.to_json())
    except Exception as e:
        print("Failed to publish message to Kafka:", e)

# Iterate over each row in the DataFrame and publish to Kafka
for index, row in df.iterrows():
    publish_to_kafka(row)

# Flush messages to Kafka (ensures all messages are delivered)
producer.flush()

# Kafka-console-consumer.bat --topic tests --bootstrap-server localhost:9092 --from-beginning
# Kafka-console-producer.bat --broker-list localhost:9092 --topic tests
# .\bin\windows\Kafka-server-start.bat .\config\server.properties
# .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties