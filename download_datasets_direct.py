import os
import pandas as pd
from huggingface_hub import hf_hub_download

def main():
    os.makedirs('data/external', exist_ok=True)
    
    # 1. Depression Reddit (Download raw CSV)
    print("Downloading Depression Reddit dataset...")
    dep_file = hf_hub_download(repo_id="mrjunos/depression-reddit-cleaned", repo_type="dataset", filename="depression_reddit_cleaned_ds.csv")
    df_dep = pd.read_csv(dep_file)
    # The CSV has columns 'text', 'labels'
    df_dep['label'] = df_dep['labels'].replace({'depression': 'depressive'})
    df_dep.to_csv('data/external/depression_reddit.csv', index=False)
    print(f"Saved depression_reddit.csv ({len(df_dep)} rows)")

    # 2. UCSD Hate Speech (Download the parquet file directly)
    print("\nDownloading UCSD Hate Speech dataset...")
    hate_file = hf_hub_download(repo_id="ucberkeley-dlab/measuring-hate-speech", repo_type="dataset", filename="data/train-00000-of-00001.parquet")
    df_hate = pd.read_parquet(hate_file)[['text', 'hate_speech_score']].dropna()
    df_hate['label'] = (df_hate['hate_speech_score'] > 0.5).map({True: 'hate_speech', False: 'normal'})
    df_hate[['text', 'label']].to_csv('data/external/hate_speech_ucsd.csv', index=False)
    print(f"Saved hate_speech_ucsd.csv ({len(df_hate)} rows)")
    
    print("\nBoth external datasets downloaded successfully!")

if __name__ == "__main__":
    main()
