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

def intent_mapping(df):
    service_mapping = {
    'book_hotel': 0,
    'find_hotel': 1,
    'book_restaurant': 2,
    'find_restaurant': 3
    }
    # Use the replace method to update the 'Service' column
    df['Intent'] = df['Intent'].apply(
        lambda x: 0 if x == 'book_hotel' 
        else (
            1 if x == 'find_hotel' 
            else (
                2 if x == 'book_restaurant' 
                else (
                3 if x == 'find_restaurant' else 4   
                )
            ) 
        )
    )

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

def extract_hotel_dialogue_acts(data):
    utterances = []
    active_intents = []

    for dialogue in data:
        for turn in dialogue['turns']:
            if 'utterance' in turn and 'frames' in turn:
                for frame in turn['frames']:
                    if 'service' in frame and 'state' in frame:
                        service = frame['service']
                        if service == 'hotel':
                            state = frame['state']
                            active_intent_value = state.get('active_intent', '')
                            if active_intent_value != 'NONE':
                                utterances.append(turn['utterance'])
                                active_intents.append(active_intent_value)

    df = pd.DataFrame({'Utterance': utterances, 'Intent': active_intents})
    df = intent_mapping(df)  # Map the intent values

    return df

def extract_restaurant_dialogue_acts(data):
    utterances = []
    active_intents = []

    for dialogue in data:
        for turn in dialogue['turns']:
            if 'utterance' in turn and 'frames' in turn:
                for frame in turn['frames']:
                    if 'service' in frame and 'state' in frame:
                        service = frame['service']
                        if service == 'restaurant':
                            state = frame['state']
                            active_intent_value = state.get('active_intent', '')
                            if active_intent_value != 'NONE':
                                utterances.append(turn['utterance'])
                                active_intents.append(active_intent_value)

    df = pd.DataFrame({'Utterance': utterances, 'Intent': active_intents})
    df = intent_mapping(df)  # Map the intent values

    return df





def check_diferent_intents(file):
    with open(file, 'r') as json_file:
        data = json.load(json_file)

    active_intents = set()
    for item in data:
        for turn in item["turns"]:
            for frame in turn.get("frames", []):  # Check if "frames" key exists
                state = frame.get("state", {})  # Check if "state" key exists
                active_intent = state.get("active_intent")  # Check if "active_intent" key exists
                if active_intent:
                    active_intents.add(active_intent)
    return list(active_intents)

