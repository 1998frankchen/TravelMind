#!/usr/bin/env python3
"""
Data preparation script for SFT training
Converts CSV format to JSON format expected by the training pipeline
"""

import pandas as pd
import json
import os
from pathlib import Path

def csv_to_json_for_sft(csv_file, output_file):
    """Convert CSV travel QA data to SFT training format"""
    df = pd.read_csv(csv_file)

    conversations = []
    for _, row in df.iterrows():
        if pd.notna(row['Question']) and pd.notna(row['Response']):
            conversation = {
                "conversations": [
                    {
                        "from": "user",
                        "value": str(row['Question']).strip()
                    },
                    {
                        "from": "assistant",
                        "value": str(row['Response']).strip()
                    }
                ]
            }
            conversations.append(conversation)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(conversations)} conversations from {csv_file} to {output_file}")
    return len(conversations)

def main():
    # Create processed data directory
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)

    # Convert training data
    train_csv = "data/sft/travel-QA/travel_QA_v2.csv"
    train_json = "data/processed/train.json"

    # Convert validation data
    val_csv = "data/sft/travel-QA/travel_QA.csv"
    val_json = "data/processed/validation.json"

    if os.path.exists(train_csv):
        train_count = csv_to_json_for_sft(train_csv, train_json)
        print(f"✅ Training data: {train_count} examples")
    else:
        print(f"❌ Training data not found: {train_csv}")

    if os.path.exists(val_csv):
        val_count = csv_to_json_for_sft(val_csv, val_json)
        print(f"✅ Validation data: {val_count} examples")
    else:
        print(f"❌ Validation data not found: {val_csv}")

    print("\n🎯 Data preparation completed!")
    print("Files created:")
    print(f"  - {train_json}")
    print(f"  - {val_json}")

if __name__ == "__main__":
    main()