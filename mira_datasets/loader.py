"""Load datasets and normalize to {input, target, meta} format."""

from typing import Dict, Any
from datasets import load_dataset, DatasetDict, Dataset
import utils.config


def load_normalized(name: str) -> DatasetDict:
    if name not in utils.config.DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(utils.config.DATASETS.keys())}")
    
    cfg = utils.config.DATASETS[name]
    mapping = cfg["field_mapping"]
    
    if cfg["hf_id"] == "json":
        files = cfg.get("data_files", {})
        if len(cfg["splits"]) == 1:
            split = cfg["splits"][0]
            file = files.get(split, list(files.values())[0] if files else None)
            ds = load_dataset("json", data_files=file)
            dataset_dict = DatasetDict({split: ds}) if isinstance(ds, Dataset) else ds
        else:
            dataset_dict = load_dataset("json", data_files=files)
    else:
        kwargs = {"path": cfg["hf_id"]}
        if cfg.get("subset"):
            kwargs["name"] = cfg["subset"]
        dataset_dict = load_dataset(**kwargs)
    
    missing = [s for s in cfg["splits"] if s not in dataset_dict]
    if missing:
        raise ValueError(f"Dataset '{name}' missing splits: {missing}. Available: {list(dataset_dict.keys())}")
    
    result = {}
    for split in cfg["splits"]:
        def norm(ex):
            inp = ex.get(mapping["input"], "")
            tgt = ex.get(mapping["target"], "")
            m = {k: v for k, v in ex.items() if k not in [mapping["input"], mapping["target"]]}
            return {
                "input": str(inp) if inp is not None else "",
                "target": str(tgt) if tgt is not None else "",
                "meta": m,
            }
        
        result[split] = dataset_dict[split].map(norm, remove_columns=dataset_dict[split].column_names)
    
    return DatasetDict(result)

