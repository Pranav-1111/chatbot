import requests

# Define the API endpoint URL
url = "http://127.0.0.1:5000/predict"

# Define the data payload to send
data = {
    "query": "Hello, how can I get support?"
}

try:
    # Send a POST request to the Flask server
    response = requests.post(url, json=data)

    # Check if the response was successful
    if response.status_code == 200:
        print("Response from chatbot:", response.json())
    else:
        print(f"Failed with status code {response.status_code}: {response.text}")

except requests.exceptions.RequestException as e:
    print("Error while making the request:", e)
