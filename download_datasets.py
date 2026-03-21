import argparse
import os
import pandas as pd
from datasets import load_dataset

def main():
    os.makedirs('data/external', exist_ok=True)
    
    print("Downloading Depression Reddit dataset...")
    # trust_remote_code=True is required for older dataset scripts
    ds_dep = load_dataset('mrjunos/depression-reddit-cleaned', trust_remote_code=True)
    df_dep = ds_dep['train'].to_pandas()
    df_dep.to_csv('data/external/depression_reddit.csv', index=False)
    print(f"✅ Saved depression_reddit.csv ({len(df_dep)} rows)")

    print("\nDownloading UCSD Hate Speech dataset...")
    ds_hate = load_dataset('ucberkeley-dlab/measuring-hate-speech', trust_remote_code=True)
    df_hate = ds_hate['train'].to_pandas()[['text', 'hate_speech_score']].dropna()
    df_hate['label'] = (df_hate['hate_speech_score'] > 0.5).map({True: 'hate_speech', False: 'normal'})
    df_hate[['text', 'label']].to_csv('data/external/hate_speech_ucsd.csv', index=False)
    print(f"✅ Saved hate_speech_ucsd.csv ({len(df_hate)} rows)")
    
    print("\n🎉 Both external datasets downloaded successfully!")
    print("You can now run: python -m src.collect.data_merger")

if __name__ == "__main__":
    main()
