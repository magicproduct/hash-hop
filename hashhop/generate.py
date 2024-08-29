import copy
import math
import random
import string
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

TASK_PROMPT = """{VARIABLE_LIST}"""

COMPLETION_PROMPT = """COMPLETION:
Hops={HOPS}

CoT={COT}

{COMPLETION}
"""


def make_random_string(length: int) -> str:
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    return "".join(alphabet[ix] for ix in np.random.choice(len(alphabet), length))


def shuffle_dict(to_shuffle: dict) -> dict:
    items = list(to_shuffle.items())
    random.shuffle(items)
    return dict(items)


@dataclass
class MultiHopSample:
    prompt: str
    """The prompt includes the shuffled hash pairs"""
    completion: str
    """This is what the model should output given the prompt"""
    targets: Dict[str, str]  # Query: GroundTruth


class MultiHopEval:
    @staticmethod
    def make_one(
        n_chars_problem: int,
        num_queries: int,
        hops: int,
        hash_pair_str_length: int,
        chain_of_thought: bool,
    ) -> MultiHopSample:
        """
        :param n_chars_problem: prompt size, in characters
        :param num_queries: number of problems in the completion
        :param hops: number of hops
        :param hash_pair_str_length: number of characters per hash
        :param chain_of_thought:
            if True, model is asked to produce H1 -> H2 -> H3.
            if False, model is asked to produce H1 -> H3
        :return: MultiHopSample
        """
        chars_per_hash_pair = (
            hash_pair_str_length * 2 + 3
        ) * hops  # abc = def has hash_pair_str_length=3 and 4 chars for equals/spaces/newline
        n_chains = math.ceil(n_chars_problem / chars_per_hash_pair)

        levels = MultiHopEval._make_levels(
            n=n_chains, hops=hops, string_length=hash_pair_str_length
        )
        lines = []
        for i, length in enumerate(levels):
            if i == len(levels) - 1:
                # Quotes indicate the end of a chain
                lines.extend([f"{k} = '{v}'" for k, v in length.items()])
            else:
                lines.extend([f"{k} = {v}" for k, v in length.items()])
        all_query_pairs = copy.deepcopy(levels[0])
        all_query_strings = {k: "" for k in all_query_pairs.keys()}
        if hops > 1:
            for i, length in enumerate(levels[1:]):
                if chain_of_thought:
                    if i == 0:
                        all_query_strings = {k: f"{v}" for k, v in all_query_pairs.items()}
                    else:
                        all_query_strings = {
                            k: f"{all_query_strings[k]} = {v}" if all_query_strings[k] != "" else v
                            for k, v in all_query_pairs.items()
                        }

                all_query_pairs = {k: length[v] for k, v in all_query_pairs.items()}
        if chain_of_thought:
            all_query_strings = {
                k: f"{all_query_strings[k]} = {v}" if all_query_strings[k] != "" else v
                for k, v in all_query_pairs.items()
            }
        else:
            all_query_strings = all_query_pairs

        random.shuffle(lines)
        shuffle_dict(all_query_strings)

        assert num_queries <= len(
            all_query_strings
        ), f"Got {num_queries} and {len(all_query_strings)}"

        completion = COMPLETION_PROMPT.format(
            COMPLETION="\n".join(
                [f"{k} = '{v}'" for k, v in list(all_query_strings.items())[:num_queries]]
            ),
            HOPS=hops,
            COT=chain_of_thought,
        )
        return MultiHopSample(
            prompt=TASK_PROMPT.format(VARIABLE_LIST="\n".join(lines)),
            completion=completion,
            targets=all_query_strings,
        )

    @staticmethod
    def _make_levels(n: int, hops: int, string_length: int) -> List[Dict[str, str]]:
        levels = [
            {
                make_random_string(length=string_length): make_random_string(length=string_length)
                for _ in range(n)
            }
        ]
        for _ in range(hops - 1):
            levels.append(
                {v: make_random_string(length=string_length) for v in levels[-1].values()}
            )
        return levels


def main() -> None:
    chars_per_token = 3
    datapoint = MultiHopEval.make_one(
        n_chars_problem=int(1_000_000 * chars_per_token),
        num_queries=5,
        hops=2,
        hash_pair_str_length=16,
        chain_of_thought=False,
    )
    print(datapoint.prompt)
    print(datapoint.completion)
    print(datapoint.targets)


if __name__ == "__main__":
    main()
