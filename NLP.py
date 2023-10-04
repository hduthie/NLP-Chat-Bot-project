import json
import pandas as pd

# Step 1: Download the JSON data or specify the path to your local JSON file
json_file_path = 'dialogues_002.json'

# Step 2: Load the JSON Data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Step 3: Extract Data and Create a DataFrame
# Initialize empty lists to store extracted data
utterances = []
services = []

# Extract data from each dialogue in the file
for dialogue in data:
    for turn in dialogue['turns']:
        if 'utterance' in turn and 'frames' in turn:
            for frame in turn['frames']:
                if 'service' in frame and 'state' in frame:
                    state = frame['state']
                    active_intent = state.get('active_intent', '')
                    if active_intent != 'NONE':
                        utterances.append(turn['utterance'])
                        services.append(frame['service'])

# Create a Pandas DataFrame
df = pd.DataFrame({'Utterance': utterances, 'Service': services})

# Step 4: Train a Predictor
# Now, you can use the 'df' DataFrame to train your predictor, for example, using machine learning algorithms.

# Example: Print the first few rows of the DataFrame
print(df.head(20))
