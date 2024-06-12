from argparse import ArgumentParser
import json
import datetime

def format_market_details(details):
    for market in details:
        if market.get("isResolved") and market.get("outcomeType") == "BINARY":
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
            out += f"Resolution: {resolution}\n"
            if resolution not in {"Yes.", "No."}:
                continue
            print(out)
        else:
            continue

parser = ArgumentParser()
parser.add_argument("market_details_path")
args = parser.parse_args()

with open(args.market_details_path) as infile:
    market_details = json.load(infile)

format_market_details(market_details)
