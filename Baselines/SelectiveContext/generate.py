import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm
from datasets import Dataset
from generator.scaledown import ScaleDownGenerator
from generator.base import BaseGenerator
from utils import load_compressed_dataset, get_dataset_path, get_metadata_path, setup_experiment


def generate_responses(config: Dict[str, Any]) -> None:
    """Main generation function - similar to compress_dataset()"""
    print("Starting response generation...")

    # Load compressed dataset, filter unprocessed samples
    dataset = load_compressed_dataset(config)
    unprocessed_indices = _filter_unprocessed_samples(dataset)

    if not unprocessed_indices: # If all samples have generated responses, return
        print("All samples already have generated responses. Nothing to process.")
        return

    # Initialize answer generator
    generator = _create_generator(config)

    # Output path
    output_path = get_dataset_path(config)

    print(f"Processing {len(unprocessed_indices)} samples...")
    responses = []
    for idx_pos, sample_idx in enumerate(tqdm(unprocessed_indices)):
        try:
            sample = dataset[sample_idx]
            
            query: str = sample['query']
            compressed_passages: list[str] = sample['passages']['passage_text']
            context = "\n".join(compressed_passages) #  Context is the context of the passages

            # Generate response
            response_data = generator.generate_response(query, context)
            responses.append(response_data)

            # Save progress periodically (every 10 samples)
            if (idx_pos + 1) % 10 == 0:
                partial_indices = unprocessed_indices[:idx_pos + 1]
                partial_responses = responses[:idx_pos + 1]
                
                updated_dataset = _update_dataset_with_responses(
                    dataset, partial_responses, partial_indices)
                
                updated_dataset.save_to_disk(output_path)
                print(f"Saved progress: {idx_pos + 1}/{len(unprocessed_indices)} samples processed")

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            # Add error response
            error_response = {
                'generated_response': '',
                'generation_metadata': {
                    'generator': generator.get_generator_name(),
                    'model': generator.model,
                    'timestamp': datetime.now().isoformat(),
                    'api_response_time': 0,
                    'generation_error': str(e)
                }
            }
            responses.append(error_response)

    # Final update and save
    if responses:
        updated_dataset = _update_dataset_with_responses(dataset, responses, unprocessed_indices)
        updated_dataset.save_to_disk(output_path)

        # Update metadata with generation info
        metadata_path = get_metadata_path(config)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            metadata['generation_info'] = {
                'generator': generator.get_generator_name(),
                'model': generator.model,
                'processed_samples': len(responses),
                'total_samples': len(dataset),
                'generation_timestamp': datetime.now().isoformat()
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Generated responses saved to: {output_path}")
        print(f"Processed {len(responses)} samples successfully")


def _create_generator(config: Dict[str, Any]) -> BaseGenerator:
    """Factory function to create appropriate generator"""
    generator_type = config['generation']['generator_type']
    if generator_type == "scaledown":
        return ScaleDownGenerator(config)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def _filter_unprocessed_samples(dataset: Dataset) -> List[int]:
    """Filter samples that don't have generated_response field"""
    unprocessed_indices = []

    for i, sample in enumerate(dataset):
        if 'generated_response' not in sample or not sample[
                'generated_response']:
            unprocessed_indices.append(i)

    print(
        f"Found {len(unprocessed_indices)} unprocessed samples out of {len(dataset)} total"
    )
    return unprocessed_indices


def _update_dataset_with_responses(dataset: Dataset, responses: List[Dict[str, Any]], update_indices: List[int]) -> Dataset:
    """Update specific indices with generated responses"""
    print("Updating dataset with generated responses...")

    updated_data = []
    response_idx = 0

    for i, sample in enumerate(dataset):
        sample_dict = dict(sample)  # Convert to dict for modification

        if i in update_indices and response_idx < len(responses):
            sample_dict.update(responses[response_idx]) # Update the sample with the response
            response_idx += 1

        updated_data.append(sample_dict)

    return Dataset.from_list(updated_data)


def main() -> None:
    config = setup_experiment("generate.py")
    
    print("Starting response generation...")
    generate_responses(config)
    print("Response generation completed!")


if __name__ == "__main__":
    main()
