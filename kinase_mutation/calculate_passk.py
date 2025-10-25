import pandas as pd
import numpy as np

def calculate_average_passk(csv_file_path, k_values=[1, 2, 4,8,16,32,64,128]):

    df = pd.read_csv(csv_file_path)
    

    successful_sequences = df[df['Success'] == True]
    failed_sequences = df[df['Success'] == False]
    
    average_passk = {}
    
    for k in k_values:
        success_within_k = 0
        
        for _, row in df.iterrows():
            if row['Success']:
                if row['Attempts_Needed'] <= k:
                    success_within_k += 1

        passk = success_within_k / len(df)
        average_passk[f'Pass@{k}'] = passk
        
        print(f"Pass@{k}: {passk:.4f} ({success_within_k}/{len(df)})")
    
    return average_passk


def main():

    dpo_file = "output_PhoQ/DPO/wildtype_sampling_results.csv"
    
    dpo_passk = calculate_average_passk(dpo_file)

    dpo_results_df = pd.DataFrame([dpo_passk])
    dpo_results_df.to_csv("./output_PhoQ/DPO/average_passk_results.csv", index=False)
        

if __name__ == "__main__":
    main()
