"""Dataset configuration for reasoning datasets."""

DATASETS = {
    "gsm8k": {
        "hf_id": "gsm8k",
        "subset": None,
        "splits": ["train", "test"],
        "field_mapping": {
            "input": "question",
            "target": "answer",
        },
    },
    "math": {
        "hf_id": "lighteval/MATH",
        "subset": None,
        "splits": ["train", "test"],
        "field_mapping": {
            "input": "problem",
            "target": "solution",
        },
    },
    "hotpotqa": {
        "hf_id": "hotpot_qa",
        "subset": "fullwiki",
        "splits": ["train", "validation"],
        "field_mapping": {
            "input": "question",
            "target": "answer",
        },
    },
    "strategyqa": {
        "hf_id": "metaeval/strategyqa",
        "subset": None,
        "splits": ["train", "validation"],
        "field_mapping": {
            "input": "question",
            "target": "answer",
        },
    },
    "piqa": {
        "hf_id": "piqa",
        "subset": None,
        "splits": ["train", "validation"],
        "field_mapping": {
            "input": "goal",
            "target": "sol1",  # sol2 goes to meta
        },
    },
    "commonsenseqa": {
        "hf_id": "commonsense_qa",
        "subset": None,
        "splits": ["train", "validation"],
        "field_mapping": {
            "input": "question",
            "target": "answerKey",
        },
    },
    "mbpp": {
        "hf_id": "mbpp",
        "subset": None,
        "splits": ["train", "test", "prompt"],
        "field_mapping": {
            "input": "text",
            "target": "code",
        },
    },
    "humaneval": {
        "hf_id": "openai_humaneval",
        "subset": None,
        "splits": ["test"],
        "field_mapping": {
            "input": "prompt",
            "target": "canonical_solution",
        },
    },
    "aime24": {
        "hf_id": "json",
        "subset": None,
        "splits": ["validation"],
        "data_files": {"validation": "data/aime24.jsonl"},
        "field_mapping": {
            "input": "input",
            "target": "target",
        },
    },
}
