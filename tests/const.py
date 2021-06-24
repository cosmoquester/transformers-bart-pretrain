import os

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
DEFAULT_SPM_MODEL = os.path.join(TEST_DIR, "..", "sp_model", "sp_model_unigram_8K.model")
