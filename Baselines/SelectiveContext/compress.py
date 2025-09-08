import os
import json
from typing import Dict, Any
from tqdm import tqdm
from datasets import load_dataset, Dataset
from compressor.selective_context import SelectiveContext
from utils import get_dataset_path, get_metadata_path, setup_experiment


def compress_dataset(config: Dict[str, Any]) -> None:
    """Compress the MS-MARCO dataset using SelectiveContext"""
    print("Loading MS-MARCO dataset...")
    dataset = load_dataset(config['dataset']['name'],
                           config['dataset']['version'],
                           split=config['dataset']['split'])
    print(f"Loaded {len(dataset)} samples")

    # Filter dataset for query_type
    dataset = dataset.filter(
        lambda x: x['query_type'] == config['dataset']['query_type'])
    dataset = dataset.select(
        range(min(config['dataset']['max_examples'], len(dataset))))
    print(f"Will process {len(dataset)} samples")

    # Initialize the compressor
    compressor = SelectiveContext(
        model_type=config['compressor']['model_type'],
        lang=config['compressor']['lang'],
        reduce_ratio=config['compressor']['reduce_ratio'],
        reduce_level=config['compressor']['reduce_level'])

    compressed_data = []

    print(f"Compressing {len(dataset)} samples...")
    for item in tqdm(dataset):
        try:
            # Extract the data
            question = item['query']
            passages = item['passages']
            passage_texts = passages['passage_text']
            full_context = " ".join(passage_texts)

            # Compress each passage individually
            compressed_passage_texts = []
            context_units_per_passage = []
            kept_masks_per_passage = []

            for passage_text in passage_texts:
                (
                    _,  # We dont compress the question
                    compressed_passage,
                    _,
                    _,
                    _,
                    _,  # We don't have answer choices (legacy code from LongBench)
                    unit_mask,
                    original_units) = compressor.run(question, passage_text)
                compressed_passage_texts.append(compressed_passage)
                context_units_per_passage.append(original_units)
                kept_masks_per_passage.append(unit_mask)

            # Reconstruct compressed context for ratio calculation
            compressed_context_full = " ".join(compressed_passage_texts)

            # Create compressed item
            compressed_item = {
                "query_id": item["query_id"],
                "query_type": item["query_type"],
                "query": item["query"],
                "answers": item["answers"],
                "wellFormedAnswers": item.get("wellFormedAnswers", []),
                "passages": {
                    "passage_text": compressed_passage_texts,
                    "is_selected": passages["is_selected"],
                    "url": passages["url"],
                    "context_unit": context_units_per_passage,
                    "kept_context_unit_mask": kept_masks_per_passage,
                },
                "original_passages": item["passages"],
                "compression_words_ratio": 
                    len(compressed_context_full.split()) / len(full_context.split()) if full_context.split() else 1.0,
                "compression_characters_ratio": 
                    len(compressed_context_full) / len(full_context) if full_context else 1.0,
                "compression_rate": config['compressor']['reduce_ratio'],
                "compression_level": config['compressor']['reduce_level'],
                "compression_error": ""
            }

            compressed_data.append(compressed_item)

        except Exception as e:
            print(f"Error processing item {item['query_id']}: {e}")
            # If compression fails, keep original data
            compressed_item = {
                "query_id": item["query_id"],
                "query_type": item["query_type"],
                "query": item["query"],
                "answers": item["answers"],
                "wellFormedAnswers": item.get("wellFormedAnswers", []),
                "passages": item["passages"],
                "original_passages": item["passages"],
                "context_unit": [],
                "kept_context_unit_mask": [],
                "compression_words_ratio": 1.0,
                "compression_characters_ratio": 1.0,
                "compression_rate": config['compressor']['reduce_ratio'],
                "compression_level": config['compressor']['reduce_level'],
                "compression_error": str(e)
            }
            compressed_data.append(compressed_item)

    # Create and save compressed dataset
    compressed_dataset = Dataset.from_list(compressed_data)
    output_path = get_dataset_path(config)
    compressed_dataset.save_to_disk(output_path)

    # Save metadata
    metadata = {
        "original_dataset": "MS-MARCO",
        "compressed_dataset": os.path.basename(output_path),
        "compressor":
            "SelectiveContext",
            "compressor_params": {
                "model_type": config['compressor']['model_type'],
                "lang": config['compressor']['lang'],
                "reduce_ratio": config['compressor']['reduce_ratio'],
                "reduce_level": config['compressor']['reduce_level']
            },
        "total_samples": len(compressed_data),
        "average_compression_words_ratio":
            sum(
                item.get("compression_words_ratio", 1.0)
                for item in compressed_data
            ) / len(compressed_data),
        "save_dir": config['save_dir'],
    }

    metadata_path = get_metadata_path(config)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Compressed dataset saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(
        f"Average compression ratio: {metadata['average_compression_words_ratio']:.3f}"
    )


def main() -> None:
    config = setup_experiment("compress.py")
    
    print("Starting dataset compression...")
    compress_dataset(config)
    print("Dataset compression completed!")


if __name__ == "__main__":
    main()
