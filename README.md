# Measurement of propensities in AI

## Usage example for annotating propensity demand intervals with AzureOpenAI
```python
import azure_utils as azutils
import annotation_utils as annutils

API_KEY = ...

# Create Azure client
client = azutils.get_client(api_key = API_KEY)

# Single annotation
annotation = annutils.PropensityAnnotation(
    propensity = "dummy example",
    system_prompt = "You're an capable system annotating upper and lower bounds on task instances. You use the following rubric for deciding how to annotate: ",
    rubric = "If the sentences has an even number of words, the upper bound is -2; otherwise the upper bound is 1. For the lower bound, you count the number of the most repeated word in the instance: if it's below 2, the lower bound is -3; otherwise, the upper bound is 1. ",
    task_prompt = "Annotate the following task: 'How would you like liking a likeable you?'",
    source = "gpt-4o_inference", # Deployment model name
    metadata = {
        "instance_id": "c0001"
    }
)

annotation.annotate(
    client = client
)
print(annotation.lower_bound, annotation.upper_bound, annotation.metadata["explanation"])
```

```python
1.0 -2.0 The sentence has an even number of words (8), setting the upper bound to -2. The most repeated word ('like') appears 3 times, setting the lower bound to 1.
```