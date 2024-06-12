import json
import datetime
import random
from argparse import ArgumentParser

def format_market_details(market):
    question = market.get("question")
    yes_probability = market.get("probability") * 100
    no_probability = (1 - market.get("probability")) * 100
    unique_bettor_count = market.get("uniqueBettorCount")
    creator_name = market.get("creatorName")
    created_time = datetime.datetime.fromtimestamp(market.get("createdTime") / 1000).strftime("%Y-%m-%d at %H:%M UTC")
    close_time = datetime.datetime.fromtimestamp(market.get("closeTime") / 1000).strftime("%Y-%m-%d at %H:%M UTC")
    text_description = market.get("textDescription")
    resolution = market.get("resolution").title() + "."
    out = ""
    out += "Manifold Markets\n\n"
    out += f"{question}\n"
    out += f"YES {yes_probability:.2f}% NO {no_probability:.2f}% "
    out += f"| {unique_bettor_count} Bettors\n"
    out += f"Creator: {creator_name}\n"
    out += f"Created: {created_time}\n"
    out += f"Closes: {close_time}\n\n"
    out += f"Description & Resolution Criteria: {text_description}\n\n"
    out += f"Resolution: {resolution}"
    return out

parser = ArgumentParser()
parser.add_argument("market_details")
parser.add_argument("market_scores")
args = parser.parse_args()

# Load market details and scores
with open(args.market_details) as f:
    market_details = json.load(f)
with open(args.market_scores) as f:
    market_scores = json.load(f)

# Filter market details to only include those whose key is in market scores
market_details = [v for v in market_details if v["id"] in market_scores]

# Create a list of tuples containing the market key, market details, and combined score
market_tuples = []
for market in market_details:
    key = market["id"]
    combined_score = sum([market_scores[key]['resolvable'], market_scores[key]['personal'], market_scores[key]['degeneracy']])
    market_tuples.append((market, combined_score))

# Sort the list of tuples by combined score, then by market key
sorted_markets = sorted(market_tuples, key=lambda x: (x[1], x[0]["id"]))

# Make train, val, and test splits
pool = [x[0] for x in sorted_markets[:10000]]
random.shuffle(pool)
train = pool[:9000]
val = pool[9000:]

with open("train.json", "w") as outfile:
    json.dump(train, outfile)

with open("val.json", "w") as outfile:
    json.dump(val, outfile)

# Test set based on resolved after December 12 2023
# Which is the SOLAR 10.7B release date
# https://huggingface.co/upstage/SOLAR-10.7B-v1.0/tree/main
test = [market for market in val if (market["createdTime"] / 1000) > 1702368000] 

with open("test.json", "w") as outfile:
    json.dump(test, outfile)

#count = 0
# Print the sorted markets
#for market, combined_score in sorted_markets:
#    print(format_market_details(market), combined_score, f"# {count}", end="\n\n\n\n")
#    count += 1
