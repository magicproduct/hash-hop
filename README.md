# HashHop Long Context Evaluation

This repository contains the code for
HashHop, [our long context architecture benchmark](https://magic.dev/blog/100m-token-context-windows).

## Installation Guide

### Prerequisites

- Git
- Python 3.9+
- [Poetry](https://python-poetry.org/docs/#installation)

### Steps

1. Clone the repository:
   ```
   git clone git@github.com:magicproduct/hash-hop.git
   cd hash-hop
   ```

2. Install dependencies:
   ```
   poetry install
   ```

## Generating Evaluation Data

The `MultiHopEval.make_one` function generates a `MultiHopSample` object which can be used for either evaluation (via
the `targets` field) or for training models on the multihop task (via the `completion` field).

### Usage Example

```python
from hashhop import MultiHopEval

CHARS_PER_TOKEN = 3
datapoint = MultiHopEval.make_one(
    n_chars_problem=int(1_000_000 * CHARS_PER_TOKEN),
    num_queries=5,
    hops=2,
    hash_pair_str_length=16,
    chain_of_thought=False,
)
print(datapoint.prompt)
print(datapoint.completion)
print(datapoint.targets)
```

### Parameters

- `n_chars_problem`: int
    - The size of the problem in characters.
- `num_queries`: int
    - The number of queries in the completion.
- `hops`: int
    - The number of hops in the reasoning chain.
- `hash_pair_str_length`: int
    - The number of characters per hash.
- `chain_of_thought`: bool
    - If True, the model is asked to produce H1 -> H2 -> H3.
    - If False, the model is asked to produce H1 -> H3.

### Output

- `prompt`: str
    - Contains the shuffled hash pairs.
- (Used for training) `completion`: str
    - The queries and targets in string format
- (Used for evaluation) `targets`: Dict[str, str]
    - Contains query-ground truth pairs in structured format
    - If chain of thought is false, will contain {H1: H3} (e.g. 'HETyxiWTFSVUYega': 'pChfybAJRUBmdAGC')
    - If chain of thought is true, will contain full chain {H1: H2 = H3} (e.g. 'KeiVcwXpnYIWLPmk': 'GmmNmICdvEErHgei =
      JhgvBFdYCnLVZBoy')

## Citation

```
@misc{magic2024hashhop,
  author = {Magic},
  title = {HashHop: Long Context Evaluation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/magicproduct/hash-hop}},
}
```

## License

[MIT](./LICENSE)
