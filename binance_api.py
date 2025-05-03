import os
import base64
import time
import requests
from urllib.parse import urlencode
from cryptography.hazmat.primitives import serialization

# SSH key path
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")


def load_private_key():
    with open(SSH_KEY_PATH, "rb") as key_file:
        private_key = serialization.load_pem_private_key(key_file.read(), password=None)
    return private_key


def sign_request(private_key, params):
    # Convert params to query string
    query_string = urlencode(params)

    # Sign the query string with Ed25519 key
    signature = private_key.sign(query_string.encode("utf-8"))

    # Return base64 encoded signature
    return base64.b64encode(signature).decode("utf-8")


def get_account_info(api_key, private_key):
    # API endpoint
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/account"
    url = f"{base_url}{endpoint}"

    # Create parameters with timestamp
    params = {"timestamp": int(time.time() * 1000)}

    # Sign the request
    signature = sign_request(private_key, params)
    params["signature"] = signature

    # Set up headers with API key
    headers = {"X-MBX-APIKEY": api_key}

    # Make the request
    response = requests.get(url, headers=headers, params=params)
    return response.json()
