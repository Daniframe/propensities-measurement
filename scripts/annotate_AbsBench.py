"""
Annotate AbstentionBench instances with propensity demand ranges (batch mode).

Run from the repository root:

    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=https://openaiazureprop.openai.azure.com/
    python -m scripts.annotate_AbsBench

See `notebooks/annotate_propensity_single_calls.ipynb` for the single-call
(non-batch) variant of the same pipeline, explained step by step.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src import annotation_utils as annutils
from src import azure_utils as azutils

# Repository root, so the script works from any working directory
ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    "instance_id", "question"
]

# Propensity name -> rubric code. The code is also the rubric filename stem
# under `rubrics/`, so every entry here must have a matching `rubrics/<code>.txt`.
DIMENSIONS = {
    "blue vs red colour bias": "BBRC",
    "introversion/extraversion": "PI",
    "risk aversion": "PRA",
    "ultracrepidarianism": "PU",
}

SYS_PROMPT = "You are an expert at analyzing questions for cognitive biases."

DEPLOYMENT = "gpt-4.1"

# gpt-5.x / o-series deployments reject `temperature`; pass None for those
TEMPERATURE = 0.0

N_SAMPLE = 500

if __name__ == "__main__":

    load_dotenv()

    # `get_full_prompt()` concatenates the pieces verbatim, so every piece has to end
    # with a newline or the next one is glued onto its last line.
    PRES_PROMPT = (ROOT / "rubrics" / "presentation.txt").read_text(encoding = "utf-8").rstrip() + "\n"

    ds = "AbstentionBench"

    df = pd.read_json(ROOT / f"data/benchmarks/{ds}/{ds}.jsonl", lines = True)
    df["instance_id"] = [f"AbstentionBench_{i}" for i in range(df.shape[0])]
    df = df.sample(min(N_SAMPLE, df.shape[0]), random_state = 42)

    client = azutils.get_client(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://openaiazureprop.openai.azure.com/"))

    for prop_name, prop_code in DIMENSIONS.items():
        ANN_PROMPT = f"The following is a rubric for determining the propensity of showing propensity towards {prop_name}:\n\n<rubric>\n"

        rubric_path = ROOT / "rubrics" / f"{prop_code}.txt"
        if not rubric_path.exists():
            raise FileNotFoundError(f"No rubric for '{prop_name}' at {rubric_path}")

        RUBRIC = rubric_path.read_text(encoding = "utf-8").rstrip() + "\n"

        anns = []

        for i, row in df.iterrows():
            ann = annutils.PropensityAnnotation(
                propensity = prop_code,
                system_prompt = SYS_PROMPT + "\n\n" + ANN_PROMPT,
                presentation_prompt = PRES_PROMPT,
                rubric = RUBRIC,
                task_prompt = row.question,
                source = DEPLOYMENT,
                metadata = {
                    "custom_id": row.instance_id
                }
            )

            anns.append(ann)

        annotations = annutils.PropAnnotationCollection(
            anns,
            error_filename = ROOT / f"results/{ds}/{prop_code}_errors.jsonl",
            output_filename = ROOT / f"results/{ds}/{prop_code}_outputs.jsonl"
        )

        annotations.annotate_batch(
            client = client,
            custom_ids = "metadata",
            annotator_temperature = TEMPERATURE,
            verbosity = 2,
            timeout = 3600 * 8,
            schema = "free",
            regex = annutils.FINAL_RANGE_PATTERN
        )

        annotations.save_csv(
            ROOT / f"data/annotations/{ds}/ft_{ds}_{prop_code}_{DEPLOYMENT}_annotated.csv",
            fields = [
                "propensity", "system_prompt", "task_prompt", "custom_id",
                "source", "lower_bound", "upper_bound", "explanation"
            ],
            index = False)
