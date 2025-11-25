"""Dataset configuration for reasoning datasets."""

DATASETS = {
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "subset": "main",
        "splits": ["train", "test"],
        "field_mapping": {
            "input": "question",
            "target": "answer",
        },
    },
    "math": {
        "hf_id": "HuggingFaceH4/MATH-500",
        "subset": None,
        "splits": ["test"],
        "field_mapping": {
            "input": "problem",
            "target": "solution",
        },
    },
    "hotpotqa": {
        "hf_id": "hotpotqa/hotpot_qa",
        "subset": "fullwiki",
        "splits": ["train", "validation"],
        "field_mapping": {
            "input": "question",
            "target": "answer",
        },
    },
    "strategyqa": {
        "hf_id": "ChilleD/StrategyQA",
        "subset": None,
        "splits": ["train", "test"],
        "field_mapping": {
            "input": "question",
            "target": "answer",
        },
    },
    "piqa": {
        "hf_id": "nthngdy/piqa",
        "subset": None,
        "splits": ["train", "validation"],
        "field_mapping": {
            "input": "goal",
            "target": "sol1",  # sol2 goes to meta
        },
    },
    "commonsenseqa": {
        "hf_id": "tau/commonsense_qa",
        "subset": None,
        "splits": ["train", "validation"],
        "field_mapping": {
            "input": "question",
            "target": "answerKey",
        },
    },
    "mbpp": {
        "hf_id": "Muennighoff/mbpp",
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
        "hf_id": "HuggingFaceH4/aime_2024",
        "subset": None,
        "splits": ["train"],
        "field_mapping": {
            "input": "problem",
            "target": "solution",
        },
    },
    "zebralogic": {
        "hf_id": "WildEval/ZebraLogic",
        "subset": "grid_mode",
        "splits": ["test"],
        "field_mapping": {
            "input": "puzzle",
            "target": "solution",
        },
    },
}
