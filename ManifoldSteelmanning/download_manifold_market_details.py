import os
import requests
import json
import time
from tqdm import tqdm

# Define the base URL for the API
base_url = "https://api.manifold.markets/v0/market/"

# Load the markets from the JSON file
with open("manifold_markets_2024_06_05.json", "r") as infile:
    markets = json.load(infile)

# Check if a saved in-progress JSON file exists
if os.path.isfile("manifold_market_details_2024_06_05.json"):
    # Load the saved in-progress JSON file
    with open("manifold_market_details_2024_06_05.json", "r") as infile:
        market_details = json.load(infile)

    # Find the index of the next market to fetch
    next_index = len(market_details)

    # Print a message indicating that the script is resuming from a saved in-progress JSON file
    print(f"Resuming from saved in-progress JSON file. Next market to fetch: {next_index}")
else:
    # Initialize an empty list to store the market details
    market_details = []

    # Print a message indicating that the script is starting from the beginning
    print("Starting from the beginning")

# Fetch the market details for each market
for i, market in enumerate(tqdm(markets[next_index:], desc="Fetching market details")):
    # Send a GET request to the API
    response = requests.get(base_url + market["id"])

    # If the request was successful, parse the JSON response
    if response.status_code == 200:
        data = json.loads(response.text)

        # Add the fetched market details to the list
        market_details.append(data)

        # Save the market details to a JSON file every 10,000 markets
        if (i + 1) % 10000 == 0:
            with open("manifold_market_details_in_progress.json", "w") as outfile:
                json.dump(market_details, outfile)
                print(f"Saved market details for {i + 1} markets to manifold_market_details_in_progress.json")

        # Add a rate limit of 0.2 seconds per request
        time.sleep(0.2)

    else:
        print(f"Request failed with status code {response.status_code}")

# Save the market details to a JSON file
with open("manifold_market_details_2024_06_05.json", "w") as outfile:
    json.dump(market_details, outfile)
    print(f"Saved market details for {len(market_details)} markets to manifold_market_details_2024_06_05.json")
