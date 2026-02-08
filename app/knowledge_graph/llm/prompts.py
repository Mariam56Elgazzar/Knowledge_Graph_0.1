# Schema + extraction instructions/prompts

ALLOWED_NODES = [
    "Concept", "Method", "Metric", "Dataset", "Result", "Challenge",
    "Author", "Organization", "Hyperparameter", "Technique",
    "Contribution", "Theory", "Process", "Task", "Baseline",
    "Advantage", "Observation", "Limitation", "FutureWork",
    "Model", "Component", "Benchmark", "Architecture", "Algorithm",
    "Publication", "System"
]

ALLOWED_RELATIONSHIPS = [
    "RELATED_TO", "USES", "CONTAINS", "COMPARED_TO", "ILLUSTRATES",
    "TRAINED_ON", "COMPARED_TO_ON", "USED_FOR", "IMPLEMENTS", "EVALUATES",
    "ACHIEVES", "ADDRESSES", "RESULTS_IN", "PART_OF", "CONTRIBUTES_TO",
    "IMPROVES", "SUPPORTS", "DEPENDS_ON", "DESCRIBED_IN", "PROPOSES",
    "OBSERVED_IN", "EXTENDS", "LIMITS", "INTRODUCES", "CITES"
]

RESEARCH_PAPER_INSTRUCTIONS = (
    "Extract ALL entities and relations. Cover every important detail: models, methods, datasets, metrics, "
    "baselines, components, architectures, results, contributions, limitations, comparisons, authors, "
    "algorithms, techniques, tasks, benchmarks, key findings, hyperparameters. "
    "Output a JSON array. Each object: head, head_type, relation, tail, tail_type. "
    "Example: [{\"head\":\"BERT\",\"head_type\":\"Model\",\"relation\":\"TRAINED_ON\",\"tail\":\"Wikipedia\",\"tail_type\":\"Dataset\"}]. "
    "Extract 30-70 relations when possible. Output ONLY the JSON array."
)

DIRECT_PROMPT = (
    "Extract ALL entities and relations. Cover models, datasets, metrics, baselines, methods, components, "
    "results, contributions, limitations, comparisons, authors, techniques.\n"
    "Each object: {\"head\":\"entity1\",\"head_type\":\"Model\",\"relation\":\"USES\",\"tail\":\"entity2\",\"tail_type\":\"Component\"}\n"
    "Node types: Model, Method, Dataset, Metric, Component, Concept, Author, Result, Baseline, Technique, "
    "Architecture, Task, Algorithm, Benchmark, Observation, Limitation, Contribution.\n"
    "Relation types: USES, CONTAINS, RELATED_TO, PART_OF, COMPARED_TO, TRAINED_ON, EVALUATES, IMPROVES, "
    "IMPLEMENTS, ACHIEVES, ADDRESSES, RESULTS_IN, PROPOSES, EXTENDS, DEPENDS_ON, SUPPORTS, ILLUSTRATES, "
    "CONTRIBUTES_TO, INTRODUCES, OBSERVED_IN, LIMITS, CITES.\n"
    "Extract 20-40 relations. Output ONLY valid JSON array (no markdown)."
)


