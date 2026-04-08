import torch
from typing import List
import pandas as pd
from transformers import RobertaTokenizer

from data.dataset_interface import DatasetInterface
from data.schemas import DataItemSchema
from utils import config_utils, data_utils
import constants
from enums import SplitRunType

config = config_utils.load_config()


class SampleDataset(DatasetInterface):

    def __init__(self, split_run_type: SplitRunType):
        super().__init__(split_run_type)
        self.load_dataframes()
        self.load_data()

    def load_dataframes(self):
        dataframe = pd.read_csv(constants.PATH_NEPAL_QUAKE_TWEETS_DATASET)
        dataframe = data_utils.preprocess_text_data(dataframe)
        self.dataframes_split = data_utils.get_dataframes_split(dataframe)

    def get_dataframe_by_split_run_type(self):
        return self.dataframes_split[self.split_run_type]

    def load_data(self):
        dataframe = self.get_dataframe_by_split_run_type()
        self.ids = list(dataframe[constants.ID_COLUMN])
        self.tweets = list(dataframe[constants.TEXT_COLUMN])
        self.labels = torch.tensor(dataframe[constants.LABEL_COLUMN], dtype=torch.long)
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", truncation=True, do_lower_case=True
        )

    def get_data_length(self) -> int:
        return len(self.tweets)

    def get_data_item(self, idx=-1) -> DataItemSchema:
        tweet_id = self.ids[idx]
        tweet_text = " ".join(str(self.tweets[idx]).split())
        label = self.labels[idx]

        text_tokenized_inputs = data_utils.parse_tokenizer_output(
            self.tokenizer(
                tweet_text,
                add_special_tokens=True,
                max_length=config[constants.FIELD_MAX_LEN_TEXT],
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )
        )

        return DataItemSchema(
            **{
                constants.FIELD_IMAGE_TENSOR: {},
                constants.FIELD_TOKENIZED_TEXT_INPUTS: text_tokenized_inputs,
                constants.FIELD_LABEL: label,
                constants.FIELD_METADATA: {constants.ID_COLUMN: tweet_id},
            }
        )
