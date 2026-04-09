import yaml
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load YAML config
with open("config/default.yaml") as f:
    config = yaml.safe_load(f)

# Replace secrets from env
config["openai_api_key"] = os.getenv("OPENAI_API_KEY")

print(config["data_path"])
print(config["openai_api_key"])
