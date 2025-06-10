import os
import json
import argparse
from tabulate import tabulate

def read_evaluation_results(results_dir):
    """Read all evaluation result files from the specified directory."""
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract relevant information
                model_name = data['model_name'].split('model/checkpoints/')[-1].split('/')[0]
                dataset = data['dataset']
                # Use num_answers if available, otherwise use num_files
                count = data.get('num_answers', data.get('num_files', 'N/A'))
                accuracy = data['accuracy']
                
                results.append({
                    'model': model_name,
                    'dataset': dataset,
                    'count': count,
                    'accuracy': f"{accuracy*100:.2f}",
                })
    
    return results

def display_results(results):
    """Display results in a formatted table."""
    if not results:
        print("No evaluation results found.")
        return
    
    # Sort results by model name and dataset
    results.sort(key=lambda x: (x['model'], x['dataset']))
    
    # Prepare table data
    headers = ['Model', 'Dataset', 'Count', 'Accuracy']
    table_data = [[r['model'], r['dataset'], r['count'], r['accuracy']] for r in results]
    
    # Print table
    print("\nEvaluation Results:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def main():
    parser = argparse.ArgumentParser(description="Display accuracy results from evaluation files")
    parser.add_argument("--results_dir", type=str, default="evaluation_results",
                        help="Directory containing evaluation result files")
    
    args = parser.parse_args()
    
    results = read_evaluation_results(args.results_dir)
    display_results(results)

if __name__ == "__main__":
    main() 