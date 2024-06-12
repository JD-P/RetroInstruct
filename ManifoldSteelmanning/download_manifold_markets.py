import requests
import json
import time
import os

# Define the base URL for the API
base_url = "https://api.manifold.markets/v0/markets"

# Define the parameters for the API request
params = {
    "limit": 1000,
    "sort": "created-time",
    "order": "desc"
}

# Initialize an empty list to store the markets
markets = []

# Fetch the markets in batches of 1,000
while True:
    # Send a GET request to the API
    response = requests.get(base_url, params=params)

    # If the request was successful, parse the JSON response
    if response.status_code == 200:
        data = json.loads(response.text)

        # If there are no more markets to fetch, break the loop
        if not data:
            break

        # Add the fetched markets to the list
        markets.extend(data)

        # Print a progress indicator every 10,000 markets
        if len(markets) % 10000 == 0:
            print(f"Fetched {len(markets)} markets...")

        # Save the markets to a JSON file every 250,000 markets
        if len(markets) % 250000 == 0:
            with open("manifold_markets_2024_06_05.json", "w") as outfile:
                json.dump(markets, outfile)
                print(f"Saved {len(markets)} markets to manifold_markets_2024_06_05.json")

        # Update the before parameter to fetch the next batch of markets
        params["before"] = data[-1]["id"]

        # Add a rate limit of 0.2 seconds per request
        time.sleep(0.2)

    else:
        print(f"Request failed with status code {response.status_code}")
        break

# Save the markets to a JSON file
with open("manifold_markets_2024_06_05.json", "w") as outfile:
    json.dump(markets, outfile)
    print(f"Saved {len(markets)} markets to manifold_markets_2024_06_05.json")
