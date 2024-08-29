import pytest

from hashhop.generate import MultiHopEval


def parse_variable_list(variable_list: str) -> dict:
    hash_dict = {}
    for hop in variable_list.split("\n"):
        if " = " in hop:
            key, value = hop.split(" = ")
            assert key not in hash_dict
            hash_dict[key] = value.removeprefix("'").removesuffix("'")
    return hash_dict


@pytest.mark.parametrize("hops", [1, 2, 6, 10])
@pytest.mark.parametrize("hash_pair_str_length", [8, 16])
@pytest.mark.parametrize("num_queries", [5])
def test_multihop_hash(hops: int, hash_pair_str_length: int, num_queries: int) -> None:
    datapoint = MultiHopEval.make_one(
        n_chars_problem=100_000,
        num_queries=num_queries,
        hops=hops,
        hash_pair_str_length=hash_pair_str_length,
        chain_of_thought=False,
    )
    hash_dict = parse_variable_list(datapoint.prompt)

    num_queries_in_prompt = 0
    for key, value in datapoint.targets.items():
        num_hops = 0
        query_key = key
        while query_key in hash_dict:
            num_hops += 1
            query_key = hash_dict[query_key]
        # Validate num hops
        assert hops == num_hops
        num_queries_in_prompt += f"{key} = '{value}'" in datapoint.completion

    assert num_queries_in_prompt == num_queries


@pytest.mark.parametrize("hops", [1, 2, 6, 10])
@pytest.mark.parametrize("hash_pair_str_length", [8, 16])
@pytest.mark.parametrize("num_queries", [5])
def test_chain_of_thought_hash(hops: int, hash_pair_str_length: int, num_queries: int) -> None:
    datapoint = MultiHopEval.make_one(
        n_chars_problem=100_000,
        num_queries=num_queries,
        hops=hops,
        hash_pair_str_length=hash_pair_str_length,
        chain_of_thought=True,
    )
    hash_dict = parse_variable_list(datapoint.prompt)

    num_queries_in_prompt = 0
    for key, value in datapoint.targets.items():
        num_hops = 0
        query_key = key
        chain_of_thought_string = f"{query_key} = '"
        while query_key in hash_dict:
            num_hops += 1
            query_key = hash_dict[query_key]
            chain_of_thought_string += f"{query_key} = "
        chain_of_thought_string = chain_of_thought_string.removesuffix(" = ")
        chain_of_thought_string += "'"
        # Validate num hops
        assert hops == num_hops
        num_queries_in_prompt += chain_of_thought_string in datapoint.completion

    assert num_queries_in_prompt == num_queries, datapoint.completion
