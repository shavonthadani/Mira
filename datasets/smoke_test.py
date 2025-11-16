"""Quick test to make sure all datasets load."""

import json
from pathlib import Path
from datasets.loader import load_normalized
import utils.config


def main():
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Dataset Smoke Test")
    print("=" * 60)
    
    for name in utils.config.DATASETS.keys():
        print(f"\n[{name}]")
        print("-" * 60)
        
        try:
            data = load_normalized(name)
            
            for split, split_data in data.items():
                size = len(split_data)
                print(f"  {split}: {size:,} examples")
                
                preview = cache_dir / f"{name}_{split}_preview.jsonl"
                with open(preview, "w", encoding="utf-8") as f:
                    for i in range(min(3, size)):
                        f.write(json.dumps(split_data[i], ensure_ascii=False) + "\n")
                
                print(f"    → Preview written to {preview}")
        
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Smoke test completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
