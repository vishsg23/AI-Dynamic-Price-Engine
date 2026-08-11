import os
import sys
import json
import time
import pandas as pd
from kafka import KafkaProducer

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(BASE_DIR)

print("==================================================")
print("        STARTING REAL-TIME KAFKA PRODUCER         ")
print("==================================================\n")

# Initialize the stream gateway connection
try:
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
except Exception as e:
    print(f" Connection Error: Could not connect to Kafka broker. {e}")
    sys.exit(1)

# Check for transaction raw logs
# NOTE: points at streaming_holdout.csv (the 15% newest rows the model has
# NEVER trained or tested on) instead of master_features.csv, so this is a
# genuine unseen-data replay, not a re-play of data the model already learned from.
data_path = os.path.join(BASE_DIR, "data", "processed", "streaming_holdout.csv")
if not os.path.exists(data_path):
    print(f" Missing transaction logs dataset at: {data_path}")
    sys.exit(1)

sales = pd.read_csv(data_path)

def publish_order_event(row):
    event = {
        'event_type': 'order',
        'transaction_id': int(row.get('transaction_id', time.time())),
        'product_id': int(row['product_id']),
        'product_name': str(row.get('product_name', 'Unknown')),
        'units_sold': int(row.get('units_sold', row.get('total_units_sold', 1))),
        'current_price': float(row['current_price']),
        'inventory_ratio': float(row.get('inventory_ratio', 1.0)),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    producer.send('orders_topic', value=event)

print(' Commencing live message push to stream pipeline...')
for idx, row in sales.iterrows():
    publish_order_event(row)
    
    if idx % 100 == 0:
        print(f' Stream status: Published {idx} events...')
        
    time.sleep(0.05)  # Simulate real-world spacing (20 events per second)

producer.flush()
print('\n Success: All real-time event payloads sent successfully!')