import pandas as pd
from src import annotation_utils as annutils
from src import azure_utils as azutils
from dotenv import load_dotenv

# CANDIDATES = [
#     "Civil Service Examination", "LSAT", "MCTACO", "SAT", "TimeDial", "TruthQuest"
# ]

COLUMNS = [
    "instance_id", "question"
]

DIMENSIONS = {
    # "blue vs red colour bias": "BR",
    "extraversion": "Ex",
    "risk aversion": "RA",
    "delay of gratification": "TD",
    "ultracrepidarianism": "Ul",
    "coopetition": "Co",
}

SYS_PROMPT = "You are an expert at analyzing questions for cognitive biases."

if __name__ == "__main__":

    with open("rubrics/presentation.txt", "r") as f:
        PRES_PROMPT = f.read()

    ds = "AbstentionBench"

    df = pd.read_json(f"data/benchmarks/{ds}/{ds}.jsonl", lines = True)
    df["instance_id"] = [f"AbstentionBench_{i}" for i in range(df.shape[0])]
    df = df.sample(500, random_state = 42)

    for prop_name, prop_code in DIMENSIONS.items():
        ANN_PROMPT = f"The following is a rubric for determining the propensity of showing propensity towards {prop_name}:\n\n<rubric>\n"

        with open(f"rubrics/{prop_code}_v1.md", encoding = "ISO-8859-1") as f:
            RUBRIC = f.read()

        anns = []

        for i, row in df.iterrows():
            ann = annutils.PropensityAnnotation(
                propensity = prop_code,
                system_prompt = SYS_PROMPT + ANN_PROMPT,
                presentation_prompt = PRES_PROMPT,
                rubric = RUBRIC,
                task_prompt = row.question,
                source = "gpt-5.1",
                metadata = {
                    "custom_id": row.instance_id
                }
            )

            anns.append(ann)

        annotations = annutils.PropAnnotationCollection(
            anns,
            error_filename = "errors.jsonl",
            output_filename = "outputs.jsonl"
        )

        client = azutils.get_client(
            api_key = "", 
            endpoint = "https://openaiazureprop.openai.azure.com/")

        annotations.annotate_batch(
            client = client,
            custom_ids = "metadata",
            verbosity = 2,
            timeout = 3600 * 8,
            schema = "free",
            regex = r"<FINAL_RANGE>\s*\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]\s*</FINAL_RANGE>"
        )

        print(annotations.error_filename)

        annotations.save_csv(f"data/annotations/{ds}/ft_{ds}_{prop_code}_GPT-5.1_annotated2.csv", fields = [
            "propensity", "system_prompt", "task_prompt", "custom_id", "source", "lower_bound", "upper_bound", "explanation"
        ], index = False)
