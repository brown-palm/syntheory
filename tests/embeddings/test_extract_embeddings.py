from embeddings.extract_embeddings import DatasetEmbeddingInformation

def test_get_shard_sizes() -> None:
    # evenly divides
    shard_sizes = DatasetEmbeddingInformation.get_shard_sizes(1_200, 300)
    assert shard_sizes == [
        300,
        300,
        300,
        300
    ]
    
    # need extra shard
    shard_sizes = DatasetEmbeddingInformation.get_shard_sizes(1_201, 300)
    assert shard_sizes == [
        300,
        300,
        300,
        300,
        1
    ]
