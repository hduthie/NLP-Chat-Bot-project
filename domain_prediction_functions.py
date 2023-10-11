import json
import pandas as pd


def load_json(file):

    with open(file, 'r') as json_file:
        data = json.load(json_file)
    
    return data


def service_mapping(df):
    service_mapping = {
    'restaurant': 0,
    'hotel': 1,
    }
    # Use the replace method to update the 'Service' column
    df['Service'] = df['Service'].apply(lambda x: 0 if x == 'restaurant' else (1 if x == 'hotel' else 2))

    return df


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
    df = service_mapping(df) # 0=restaurant, 1=hotel, 2=others

    return df


