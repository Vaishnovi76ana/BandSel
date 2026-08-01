import json
import os

def main():
    input_file = 'optimal_predictions.jsonl'
    output_file = 'finetuning_dataset.json'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    dataset = []
    
    print(f"Reading from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if item.get('accuracy') is True:
                        question = item['messages'][0]['content']
                        compression_ratio = item['optimal_compression_ratio']
                        if compression_ratio < 1:
                            input_data = f"{question}<|eot_id|>{compression_ratio}<|eot_id|>"
                        else:
                            input_data = question
                        output_data = item['model_output']
                        
                        entry = {
                            "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
                            "input": input_data,
                            "output": output_data
                        }
                        dataset.append(entry)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
                except KeyError as e:
                    print(f"Skipping item due to missing key: {e}")

        print(f"Found {len(dataset)} items with accuracy: true.")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4)
            
        print(f"Successfully wrote {len(dataset)} items to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
