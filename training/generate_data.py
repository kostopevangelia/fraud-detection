# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:36:57 2025

@author: kostopevangelia

Data Generation
"""

import logging

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
import numpy as np


## We create helper lists with some values that we'll randomly pick for the dataset

# Bins are the first six digits of card numbers
card_bins = ['400005', '424242', '510510', '601100', '352800']

# Currencies the system might support
currencies = ['USD', 'EUR', 'GBP']

# Payment types 
payment_types = ['card', 'iban']

# Transaction types
transaction_types = ['PAYMENT', 'WITHDRAWAL', 'TRANSFER']

# Fake user IDs
user_ids = [f'USER{i}' for i in range(1000)]


## Function that generates a single transaction
def generate_transaction(is_fraud=False):
    card_bin = random.choice(card_bins);
    card_suffix = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    card_number = card_bin + card_suffix
    
    amount = round(random.uniform(1, 5000), 2)
    if is_fraud:
        amount *= random.uniform(1.5, 3.0) # That is because fraud often have higher ammounts
        
    currency = random.choice(currencies)
    payment_type = random.choice(payment_types)
    transaction_type = random.choice(transaction_types)
    user_id = random.choice(user_ids)
    
    # We put random times in July 2025
    random_days = random.randint(0, 31) 
    random_seconds = random.randint(0, 86400)
    timestamp = datetime(2025, 7, 1) + timedelta(days=random_days, seconds=random_seconds)
    timestamp = timestamp.isoformat()

    return {
        'transactionId': str(uuid.uuid4()),
        'cardNumber': card_number,
        'amount': round(amount, 2),
        'currency': currency,
        'paymentType': payment_type,
        'transactionType': transaction_type,
        'userId': user_id,
        'timestamp': timestamp,
        'fraud': int(is_fraud)
    }


# Generate legit and fraud transactions

data = []

# 5000 legit transactions
for _ in range(5000):
    data.append(generate_transaction(is_fraud=False))
    
for _ in range(500):
    data.append(generate_transaction(is_fraud=True))
    
# Shuffle and convert to DataFrame
random.shuffle(data)
df = pd.DataFrame(data)


# Save
df.to_csv("custom_transactions.csv", index=False)
logger.info("Generated custom_transactions.csv with %d rows", len(df))