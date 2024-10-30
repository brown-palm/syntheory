import argparse
from util import use_770_permissions
from embeddings.extract_embeddings import extract_shard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder_name", type=str, required=True)
    parser.add_argument("--dataset_shard", type=int, required=True)
    parser.add_argument("--model_config_checksum", type=str, required=True)
    args = parser.parse_args()

    with use_770_permissions():
        extract_shard(
            args.dataset_folder_name,
            int(args.dataset_shard),
            args.model_config_checksum,
        )
