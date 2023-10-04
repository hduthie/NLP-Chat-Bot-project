# import json

# # Load the JSON data into a Python dictionary
# with open('dialogues_002.json', 'r') as json_file:
#     data = json.load(json_file)

# # Create an empty dictionary to store the dialogues by their IDs
# dialogues_by_id = {}

# # Iterate through the list of dialogues and add them to the dictionary
# for dialogue in data:
#     dialogue_id = dialogue["dialogue_id"]
#     dialogues_by_id[dialogue_id] = dialogue

# # Define the desired dialogue ID
# desired_dialogue_id = "SNG0771.json"  # Replace with the dialogue ID you want to access

# # Access the desired dialogue by its ID from the dictionary
# desired_dialogue = dialogues_by_id.get(desired_dialogue_id)

# # Check if the dialogue was found
# if desired_dialogue:
#     # Access the first turn in the desired dialogue
#     first_turn = desired_dialogue['turns'][0]

#     # Extract the "utterance" field from the first turn
#     utterance = first_turn['utterance']

#     # Print the utterance
#     print("Utterance:", utterance)
# else:
#     print(f"Dialogue with ID '{desired_dialogue_id}' not found in the JSON data.")



# print(dialogues_by_id["SNG0771.json"])

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
