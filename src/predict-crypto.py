from google.colab import drive
drive.mount('/content/gdrive', force_remount="True")

#uploaded = files.upload()


import sqlite3

print("test")
conn = sqlite3.connect('/content/gdrive/MyDrive/Programs/AI-ML/predict-crypto/db15.sqlite')
#conn = sqlite3.connect('/drive/MyDrive/Programs/AI-ML/predict-crypto/db14.sqlite')


cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS ecoins(
    id INTEGER PRIMARY KEY,
    date VARCHAR(255) NOT NULL,
    price FLOAT NOT NULL
)
''')

conn.commit()

#books_data = [
#    ("To Kill a Mockingbird", "Harper Lee", 1960),
#    ("1984", "George Orwell", 1949),
#    ("The Great Gatsby", "F. Scott Fitzgerald", 1925)
#]

bc_data = [ ("01/21/21", 55)]

#cursor.executemany('''
#INSERT INTO bitcoins (date, price) VALUES (?, ?)
#''', bc_data)

conn.commit()

cursor.execute("SELECT * FROM ecoins")
bc = cursor.fetchall()

for b in bc:
    print(b)

cursor.execute("SELECT COUNT(*) FROM ecoins")
bc = cursor.fetchall()

print ( "count: ")
for b in bc:
    print(b)


## -----------------------

import requests
import pandas as pd
from datetime import datetime

# Define the URL to fetch data
url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
params = {
    'vs_currency': 'usd',
    'days': '30',  # Fetch data for the last 30 days
    'interval': 'daily'
}

# Fetch the data from the API
response = requests.get(url, params=params)
data = response.json()

# Process the data
prices = data['prices']
market_caps = data['market_caps']
total_volumes = data['total_volumes']

# Convert timestamps to readable dates and create a list of dictionaries
processed_data = []
for i in range(len(prices)):
    processed_data.append({
        'date': datetime.utcfromtimestamp(prices[i][0] / 1000).strftime('%Y-%m-%d'),
        'price': prices[i][1],
        'market_cap': market_caps[i][1],
        'total_volume': total_volumes[i][1]
    })

# Create a DataFrame
df = pd.DataFrame(processed_data)

# Save the DataFrame to a CSV file
csv_filename = 'bitcoin_data.csv'
df.to_csv(csv_filename, index=False)

bc_data = [ ("01/21/21", 55)]

the_data = '01/21/21'

qr = "SELECT * FROM ecoins where date = '" + the_data + "'"

cursor.execute(qr)
bc = cursor.fetchall()
print ("--------------")
print ( qr )
print ( bc)
print ("-------------")
#cursor.executemany('''SELECT IF EXISTS ( SELECT * FROM ECOINS WHERE date = (the_date) ) ''', the_data

print ( df.to_string() )

for x in df.values:
 # print ( x[0] )
 # print ( x[1] )
 # print ("---")
 # bc_data = [ x[0], x[1] ]
  #bc_data = [ ("01/21/21", 55)]

  date = x[0]
  price = x[1]

  bc_data = [ (date, price) ]

  cursor.executemany('''
  INSERT INTO ecoins (date, price) VALUES (?, ?)
  ''', bc_data)

  conn.commit()

#  conn.close()
#DELETE FROM customers
#WHERE customer_id IN (
#    SELECT customer_id
#    FROM customers
#    GROUP BY customer_id
#    HAVING COUNT(*) > 1
#);

cursor.execute('''
WITH cte AS (
    SELECT
        id,
        date,
        ROW_NUMBER() OVER (PARTITION BY date ORDER BY id) as row_num
    FROM
        ecoins
)
DELETE FROM ecoins
WHERE id IN (
    SELECT id
    FROM cte
    WHERE row_num > 1
)''')
#conn.commit()


#files.download('db12.sqlite')
#conn.close()


### write to db
#for x in range(0, len(df)):
#  print ( df.iloc(x) )


#print(f"Data has been saved to {csv_filename}")
