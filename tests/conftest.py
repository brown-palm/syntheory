import pytest
import transformers

@pytest.fixture(scope='session', autouse=True)
def set_hf_transformers_seed() -> None:
    transformers.set_seed(42)