


def evaluate_coin(coin_choice):

    import sqlite3
    conn = sqlite3.connect('db15.sqlite')

    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ''' + coin_choice + '''(
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

    the_string = "SELECT * FROM " + coin_choice
    cursor.execute(the_string)
    bc = cursor.fetchall()

    coin_data = bc

    print ("select-------")
    for b in bc:
        print(b)

    the_string = "SELECT COUNT(*) FROM " + coin_choice
    cursor.execute(the_string)
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
    csv_filename = coin_choice + '_data.csv'
    df.to_csv(csv_filename, index=False)

    bc_data = [ ("01/21/21", 55)]

    the_data = '01/21/21'

    qr = "SELECT * FROM " + coin_choice + " where date = '" + the_data + "'"

    cursor.execute(qr)
    bc = cursor.fetchall()
    print ("qr--------------")
    print ( qr )
    print ("bc---------")
    print ( bc)
    print ("-------------")
    #cursor.executemany('''SELECT IF EXISTS ( SELECT * FROM dogecoin WHERE date = (the_date) ) ''', the_data

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

      the_string = "INSERT INTO " + coin_choice + " (date, price) VALUES (?, ?) "

      cursor.executemany( the_string, bc_data )

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
            dogecoin
    )
    DELETE FROM dogecoin
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

# Just do your computation in a cell
    x = 42
    x  # The value we want to capture



    import requests
    import pandas as pd
    from datetime import datetime
    import statistics

    # Define the URL to fetch data
    url = 'https://api.coingecko.com/api/v3/coins/' + coin_choice + '/market_chart'
 
    #url = 'https://api.coingecko.com/api/v3/coins/litecoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '365',  # Fetch data for the last 30 days
        'interval': 'daily'
    }


    # Fetch the data from the API
    response = requests.get(url, params=params)
    data = response.json()
    print ('********')
    print (url)
    print ( data )
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
        '    total_volume': total_volumes[i][1]
        })

    # Create a DataFrame
    df = pd.DataFrame(processed_data)

    # Save the DataFrame to a CSV file
    csv_filename = coin_choice + '_data.csv'
    df.to_csv(csv_filename, index=False)

#df = df.resample('date').first()

#print(f"Data has been saved to {csv_filename}")

    df['date'] = pd.to_datetime(df['date'])   
    df2 = df.groupby(df['date'].dt.to_period('W')).apply(lambda x: x.loc[x['date'].idxmax()]).reset_index(drop=True)


    df3 = df2[['date', 'price']]

##----------------------------

    df5 = pd.DataFrame(columns=['date','price'])


#df5[0]= ['11-11-11', 100]

#count = 0
#for x in coin_data:

### if want to add???????????
#for x in range(1,5):
#    df3.loc[len(df3.index)] = ['11-11-11', 89]


    print ('ùuuuuuu')

    print ( df3 )

    print ( x )

##----------

    df3['moving_avg'] = 0
    df3['distance_from_avg'] = 0
    df3['sd'] = 0

    tempsum = 0
    tempavg = 0
    counttest = 0
    std_list = []
    std_current = 0
    sdt_distance = 0
    std = 0


    for x in range( 12, len(df3)):
      for y in range( x-12, x):
        tempsum += df3['price'][y]
        counttest += 1
        std_list.append(df3['price'][y])
      tempavg = tempsum / 12

      df3.at[x, 'moving_avg'] = tempavg
      counttest = 0
      tempsum = 0
      df3.at[x, 'distance_from_avg'] = df3['price'][x] - tempavg

      std_current = statistics.stdev( std_list )
#  print ( std_current )
      std_distance = df3['price'][x ] - tempavg

      std = std_distance / std_current

      df3.at[x, 'sd'] = std

    df_ai_table = pd.DataFrame(columns = ['sd_12', 'sd_9', 'sd_6', 'sd_3','sd_now'])

    sd_12 = 0
    sd_9 = 0
    sd_6 = 0
    sd_3 = 0
    sd_now = 0
    temp_pos = 0


    for x in range( 12, len(df3)):
      temp_pos = x - 12
      df3.at[x, 'sd_12'] = df3['sd'][temp_pos]
      temp_pos = x - 9
      df3.at[x, 'sd_9'] = df3['sd'][temp_pos]
      temp_pos = x - 6
      df3.at[x, 'sd_6'] = df3['sd'][temp_pos]
      temp_pos = x - 3
      df3.at[x, 'sd_3'] = df3['sd'][temp_pos]
      temp_pos = x
      df3.at[x, 'sd_now'] = df3['sd'][temp_pos]

#print ( df_ai_table )


## get last row
    last_row = df3.iloc[-1]
    print ("last row ----------")
    print ( last_row )
    print ( "-000000000000000-")

    import torch
    import torch.nn as nn
    import torch.optim as optim

# Define the neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(4, 10)  # First fully connected layer (6 inputs to 10 hidden units)
            self.fc2 = nn.Linear(10, 4)  # Second fully connected layer (10 hidden units to 5 hidden units)
            self.fc3 = nn.Linear(4, 1)   # Third fully connected layer (5 hidden units to 1 output)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

# Initialize the neural network, loss function, and optimizer
    net = SimpleNet()
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.SGD(net.parameters(), lr=0.01)  # Stochastic Gradient Descent


# Example data (6 inputs and 1 float output)
    inputs = torch.tensor([[0.5, -0.2, 0.1, 0.4],
                           [0.1, -0.4, 0.2, 0.1],
                           [0.3, -0.1, 0.4, 0.2]], dtype=torch.float32)


#tens = torch.cat((inputs, inputs), 0)

    targets = torch.tensor([[0.7],
                            [0.3],
                            [0.5]], dtype=torch.float32)


##----------------


##----------------



# Training loop
    num_epochs = 10000
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradient buffers
        outputs = net(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the network with a new input
    test_input = torch.tensor([[0.3, -0.1, 0.4, 0.2]], dtype=torch.float32)
    test_output = net(test_input)
#print('Test output:', test_output.item())



#for x in range( len(df3)):
#    print ( df3['price'][x] )


    inputs2 = torch.tensor([[]])
# define tensors
    tens_1 = torch.Tensor([[11, 12, 13], [14, 15, 16]])
    tens_2 = torch.Tensor([[17, 18, 19], [20, 21, 22]])

    inputs2 = torch.tensor([[0.5, -0.2, 0.1, 0.4],
                           [0.1, -0.4, 0.2, 0.1],
                           [0.3, -0.1, 0.4, 0.2]], dtype=torch.float32)


    targets2 = torch.tensor([[0.7],
                            [0.3],
                            [0.5]], dtype=torch.float32)



    temp_list = [[]]
    temp_out = [[]]
    for x in range ( 12, len(df3 )):
        temp_list = []
        temp_out = []
        temp_list.append ( df3['sd_12'][x])
        temp_list.append ( df3['sd_9'][x])
        temp_list.append ( df3['sd_6'][x])
        temp_list.append ( df3['sd_3'][x])
        temp_out.append( df3['sd_now'][x] )

        temp_list_output = torch.tensor([temp_list], dtype=torch.float32)
        temp_out = torch.tensor([temp_out], dtype=torch.float32)
        inputs2 = torch.cat((inputs2, temp_list_output),0)
        targets2 = torch.cat((targets2, temp_out),0)


### get last value



# Training loop
    num_epochs = 100000
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradient buffers
        outputs2 = net(inputs2)  # Forward pass
        loss = criterion(outputs2, targets2)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

  #  if (epoch+1) % 100 == 0:
  #      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Test the network with a new input
    test_input = torch.tensor([[last_row[5],  last_row[6], last_row[7], last_row[8] ]], dtype=torch.float32)
    test_output = net(test_input)
    print('Test output:', test_output.item())

    print ( last_row )

    print ( df3.to_string() )

    return_value = 10

 

# This notebook produces a value
    value_to_share = 12345

    sd_now = df3['sd'][temp_pos]

 

    print ( test_output.item() )
    print ( df3['sd'][temp_pos] )

    print('Test output:', test_output.item())
    
    return str(coin_choice), str(sd_now), str(test_output.item())
    
v1a, v2a, v3a = evaluate_coin('bitcoin')
v1b, v2b, v3b = evaluate_coin('litecoin')
v1c, v2c, v3c = evaluate_coin('ethereum')
v1d, v2d, v3d = evaluate_coin('dogecoin')

print ( v1a, 'current sd: ', v2a, 'AI predicted sd( where should be: ', v3a )
print ( v1b, 'current sd: ', v2b, 'AI predicted sd( where should be: ', v3b )
print ( v1c, 'current sd: ', v2c, 'AI predicted sd( where should be: ', v3c )
print ( v1d, 'current sd: ', v2d, 'AI predicted sd( where should be: ', v3d )
 









