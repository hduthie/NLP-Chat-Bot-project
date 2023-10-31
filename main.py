import pandas as pd
from domain_prediction_functions import load_json, utterances_domain_to_df,utterances_activeintent_to_df



if __name__ == "__main__":

    # Step 1: Upload the JSON data
    json_file_path = 'dialogues_002.json'
    data = load_json(json_file_path)

    # Step 2: Extract Data and Create a DataFrame
    df = utterances_domain_to_df(data)

    # Step 3: Print first rows
    #print(df.head(20))

    value_counts = df['Service'].value_counts()
    total_count = len(df)
    percentages = (value_counts / total_count) * 100

    # Print the results
    for value, count in value_counts.items():
        percentage = percentages[value]
        #print(f'Value {value}: Count {count}, Percentage {percentage:.2f}%')

    df.to_csv('utterances_domain.csv', index=False)

    df = utterances_activeintent_to_df(data)
    df.to_csv('utterances_activeintent.csv', index=False)


    for dialogue in data:
        dialogue_id = dialogue['dialogue_id']
        for turn in dialogue['turns']:
            if turn['speaker'] == 'USER':
                utterance = turn['utterance']
                utterance_list = []
                slot_values = turn['frames'][0]['state']['slot_values']  # Assuming only one frame per turn
                words = utterance.split()
                for word in words:
                    found_slot = False
                    for slot, values in slot_values.items():
                        if any(word.lower() in value.lower() for value in values):
                            utterance_list.append(slot)
                            found_slot = True
                            break
                    if not found_slot:
                        utterance_list.append('0')
                print(f"Dialog ID: {dialogue_id}, Utterance: {utterance}, Utterance List: {utterance_list}")

