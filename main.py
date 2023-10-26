import pandas as pd
from domain_prediction_functions import load_json, utterances_domain_to_df,utterances_activeintent_to_df



if __name__ == "__main__":

    # Step 1: Upload the JSON data
    json_file_path = 'dialogues_002.json'
    data = load_json(json_file_path)

    # Step 2: Extract Data and Create a DataFrame
    df = utterances_domain_to_df(data)

    # Step 3: Print first rows
    print(df.head(20))

    value_counts = df['Service'].value_counts()
    total_count = len(df)
    percentages = (value_counts / total_count) * 100

    # Print the results
    for value, count in value_counts.items():
        percentage = percentages[value]
        print(f'Value {value}: Count {count}, Percentage {percentage:.2f}%')

    df.to_csv('utterances_domain.csv', index=False)

    df = utterances_activeintent_to_df(data)
    df.to_csv('utterances_activeintent.csv', index=False)

    # Step 4: Train a Predictor
    # Now, you can use the 'df' DataFrame to train your predictor, for example, using machine learning algorithms.
