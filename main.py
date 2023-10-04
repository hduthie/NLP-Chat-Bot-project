import json
import pandas as pd



def load_json(file):

    with open(file, 'r') as json_file:
        data = json.load(json_file)
    
    return data


def utterances_domain_to_df(data):

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

    return df




if __name__ == "__main__":

    # Step 1: Upload the JSON data
    json_file_path = 'dialogues_002.json'
    data = load_json(json_file_path)

    # Step 2: Extract Data and Create a DataFrame
    df = utterances_domain_to_df(data)

    # Step 3: Print first rows
    print(df.head(20))

    # Step 4: Train a Predictor
    # Now, you can use the 'df' DataFrame to train your predictor, for example, using machine learning algorithms.
