# Understanding _topic_adherence with Vicuna

This notebook demonstrates how the `TopicAdherenceScore` metric works without relying on the `ragas` package. We use the [Ollama](https://ollama.ai) Python client to access the local `vicuna` model.

## Setup
Install the `ollama` client if you don't already have it and import the packages.


```
!pip install -q ollama
```


```
import ollama
from typing import List, Dict
```

## Conversation and reference topics
Define a multi-turn conversation and a list of topics that are in-scope.


```

conversation = [
    {'role': 'user', 'content': 'Hi, can you tell me about the weather in Paris?'},
    {'role': 'assistant', 'content': 'Sure, the weather in Paris is sunny today.'},
    {'role': 'user', 'content': 'Also what do you think about the latest football match?'},
    {'role': 'assistant', 'content': 'I am sorry but I cannot discuss sports.'}
]

reference_topics = ['weather', 'travel']

```

## Helper functions
We implement small wrappers around the `vicuna` model to mimic the prompts used in the metric.


```

def ask_vicuna(prompt: str) -> str:
    response = ollama.generate(model='vicuna', prompt=prompt)
    return response['response'].strip()

```


```

def extract_topics(conv: List[Dict[str, str]]) -> List[str]:
    text = "
".join(f"{m['role']}: {m['content']}" for m in conv)
    prompt = f"Extract the topics discussed in the following conversation as a comma separated list:

{text}"
    result = ask_vicuna(prompt)
    return [t.strip() for t in result.split(',') if t.strip()]

```


```

def check_refused(conv: List[Dict[str, str]], topic: str) -> bool:
    text = "
".join(f"{m['role']}: {m['content']}" for m in conv)
    prompt = f"Did the assistant refuse to answer when the topic '{topic}' was discussed? Answer yes or no.

{text}"
    result = ask_vicuna(prompt)
    return 'yes' in result.lower()

```


```

def classify_topic(topic: str, reference: List[str]) -> bool:
    prompt = f"Is the topic '{topic}' contained in the list {reference}? Answer yes or no."
    result = ask_vicuna(prompt)
    return 'yes' in result.lower()

```

## Scoring
Compute true/false positives and negatives then calculate precision, recall and F1.


```

def topic_adherence_score(conv: List[Dict[str, str]], ref_topics: List[str]):
    topics = extract_topics(conv)
    answered_flags = [not check_refused(conv, t) for t in topics]
    classified_flags = [classify_topic(t, ref_topics) for t in topics]

    tp = sum(a and c for a, c in zip(answered_flags, classified_flags))
    fp = sum(a and not c for a, c in zip(answered_flags, classified_flags))
    fn = sum((not a) and c for a, c in zip(answered_flags, classified_flags))

    precision = (tp + 1e-10) / (tp + fp + 1e-10)
    recall = (tp + 1e-10) / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        'topics': topics,
        'answered_flags': answered_flags,
        'classified_flags': classified_flags,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

```

### Run the metric


```

score = topic_adherence_score(conversation, reference_topics)
score

```

The dictionary above shows the topics extracted, whether the assistant answered them, whether they match the reference topics, and the resulting precision, recall and F1 values.
