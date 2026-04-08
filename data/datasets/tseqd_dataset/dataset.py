import torch
from typing import List, Dict
import pandas as pd
from transformers import RobertaTokenizer
from PIL import Image, ImageFile
from torchvision import transforms
from sklearn.model_selection import train_test_split

from data.dataset_interface import DatasetInterface
from data.schemas import DataItemSchema, ImageTensorsSchema, TokenizedTextInputsSchema
from utils import config_utils, data_utils, caption_utils
import constants
from enums import SplitRunType
from topic_modelling.tseqd import get_topic_model as get_tseqd_topic_model

import os

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_COL_TO_USE = "text_info_type"




class TSEQDDataset(DatasetInterface):
    def __init__(
        self,
        split_run_type: SplitRunType,
        tokenizer=caption_utils.get_default_tokenizer(),
    ) -> None:
        super().__init__(split_run_type)

        self._config = config_utils.load_config()

        assert self._config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_TSEQD_DATASET
        self.dataframes_split = self._load_dataframes_split()
        self.dataframe = self._load_dataframe()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # for resnet
                # transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275)), # for clip
            ]
        )
        self.tokenizer = tokenizer
        self.analyse_dataframe()
        self.topics_tokenized_inputs = self._get_topics_tokenized_inputs()
        self.topics = TSEQDDataset.get_topics()

    def analyse_dataframe(self):
        print()
        print(self.split_run_type)
        print("size:", self.dataframe.shape)
        print(self.dataframe.head())
        print()

    def stratified_sample(self, df, sample_fraction=0.01):
        # Get rows for each label in class_a
        class_a_rows = [
            df[df[LABEL_COL_TO_USE] == label].sample(n=1, random_state=42)
            for label in constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES
            if label in df[LABEL_COL_TO_USE].values
        ]
        class_a_rows_df = pd.concat(class_a_rows).drop_duplicates()

        # Combine rows from both classes
        label_rows_df = pd.concat([class_a_rows_df]).drop_duplicates()

        # Determine how many more rows to sample
        required_rows = int(len(df) * sample_fraction)
        additional_rows = required_rows - len(label_rows_df)

        # Randomly sample additional rows excluding the already selected ones
        remaining_df = df.drop(label_rows_df.index)
        additional_rows_df = remaining_df.sample(
            n=max(additional_rows, 0), random_state=42
        )

        # Combine selected rows
        result_df = pd.concat([label_rows_df, additional_rows_df]).drop_duplicates()
        return result_df
    
    def _keep_selected_labels(self, df):
        keep_labels = [3, 4, 7, 8]
        filtered_df = df[df[LABEL_COL_TO_USE].isin(keep_labels)].copy()
        label_mapping = {old_label: new_index for new_index, old_label in enumerate(keep_labels)}
        filtered_df[LABEL_COL_TO_USE] = filtered_df[LABEL_COL_TO_USE].map(label_mapping)
        return filtered_df
    
    def _split_dataframe(self, df, train_size=0.8, valid_size=0.1, test_size=0.1, seed=42):
        assert abs(train_size + valid_size + test_size - 1.0) < 1e-6, "Splits must sum to 1"
        label_col = LABEL_COL_TO_USE

        # First split: Train and temp (valid+test)
        df_train, df_temp = train_test_split(
            df,
            train_size=train_size,
            stratify=df[label_col],
            random_state=seed
        )

        # Remaining size proportions
        remaining = valid_size + test_size
        valid_prop = valid_size / remaining
        test_prop = test_size / remaining

        # Second split: Validation and Test
        df_valid, df_test = train_test_split(
            df_temp,
            test_size=test_prop,
            stratify=df_temp[label_col],
            random_state=seed
        )

        return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)

    def _load_dataframes_split(self) -> Dict[SplitRunType, pd.DataFrame]:
        df = pd.read_csv(
           constants.PATH_TSEQD_DATASET_TSV_PATH,
            sep="\t",
        )
        df = self._keep_selected_labels(df)

        print('unique values',sorted(df[LABEL_COL_TO_USE].unique()))

        class_counts = df[LABEL_COL_TO_USE].value_counts().sort_index()
        print('df', class_counts)

        train_df, validation_df, test_df = self._split_dataframe(df)

        train_df = train_df[
            train_df[LABEL_COL_TO_USE].isin(
                constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES
            )
        ]
        validation_df = validation_df[
            validation_df[LABEL_COL_TO_USE].isin(
                constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES
            )
        ]
        test_df = test_df[
            test_df[LABEL_COL_TO_USE].isin(
                constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES
            )
        ]

        class_counts = train_df[LABEL_COL_TO_USE].value_counts().sort_index()
        print('train_df', class_counts)

        class_counts = validation_df[LABEL_COL_TO_USE].value_counts().sort_index()
        print('validation_df', class_counts)

        class_counts = test_df[LABEL_COL_TO_USE].value_counts().sort_index()
        print('test_df', class_counts)

        # train_df = train_df[train_df["label_text_image"] == "Positive"]
        # validation_df = validation_df[validation_df["label_text_image"] == "Positive"]
        # test_df = test_df[test_df["label_text_image"] == "Positive"]

        # train_df = self.stratified_sample(train_df)
        # validation_df = self.stratified_sample(validation_df)
        # test_df = self.stratified_sample(test_df)

        assert sorted(list(set(train_df[LABEL_COL_TO_USE].tolist()))) == sorted(
            list(set(constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES))
        ), sorted(list(set(train_df[LABEL_COL_TO_USE].tolist())))

        assert sorted(list(set(validation_df[LABEL_COL_TO_USE].tolist()))) == sorted(
            list(set(constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES))
        ), sorted(list(set(validation_df[LABEL_COL_TO_USE].tolist())))

        assert sorted(list(set(test_df[LABEL_COL_TO_USE].tolist()))) == sorted(
            list(set(constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES))
        ), sorted(list(set(test_df[LABEL_COL_TO_USE].tolist())))

        dataframes_split = {
            SplitRunType.TRAIN: train_df,
            SplitRunType.VALIDATION: validation_df,
            SplitRunType.TEST: test_df,
        }
        return dataframes_split

    def _get_dataframe_by_split_run_type(self) -> pd.DataFrame:
        return self.dataframes_split[self.split_run_type]

    def _load_dataframe(self) -> pd.DataFrame:
        dataframe = self._get_dataframe_by_split_run_type()
        return dataframe

    def get_data_length(self) -> int:
        return len(self.dataframe)

    def _get_topics_tokenized_inputs(self) -> List[TokenizedTextInputsSchema]:
        if self._config.get("train_bert_topic"):
            return []

        topics = TSEQDDataset.get_topics()

        topics_tokenized_inputs = [
            data_utils.parse_tokenizer_output(
                self.tokenizer(
                    topic["topic"],
                    add_special_tokens=True,
                    max_length=3,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )
            ).model_dump()
            for topic in topics
        ]
        return topics_tokenized_inputs

    def get_data_item(self, idx=-1) -> DataItemSchema:
        df_image_path = self.dataframe.iloc[idx]["image_path"]
        image_file_name = os.path.basename(df_image_path)   
        image_path = f"/home/shahid/4.TSEQD/Final_Images_TSEQD/{image_file_name}"

        image_id = f"{self.split_run_type}_{image_file_name}"

        image = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)

        caption = self.dataframe.iloc[idx]["text"]
        caption = caption_utils.clean_text(caption)
        caption_text_tokenized_inputs = data_utils.parse_tokenizer_output(
            self.tokenizer(
                caption,
                add_special_tokens=True,
                max_length=self._config.get(constants.FIELD_MAX_LEN_TEXT),
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )
        )

        informativeness_label = self.dataframe.iloc[idx][LABEL_COL_TO_USE]

        assert informativeness_label >= 0 and informativeness_label <= len(
            constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES
        )

        # Compute mean of non-NaN values
        nan_mask = torch.isnan(image_tensor)
        mean_value = torch.nanmean(image_tensor)  # Computes mean while ignoring NaNs

        # Replace NaNs with mean
        image_tensor[nan_mask] = mean_value

        data_item = DataItemSchema(
            image_tensors={
                constants.FIELD_INFORMATIVENESS_CATEGORY: ImageTensorsSchema(
                    **{constants.FIELD_RGB_PIXELS_TENSOR: image_tensor}
                )
            },
            tokenized_text_inputs={
                constants.FIELD_INFORMATIVENESS_CATEGORY: caption_text_tokenized_inputs
            },
            topics_tokenized_inputs=self.topics_tokenized_inputs,
            label={
                constants.FIELD_INFORMATIVENESS_CATEGORY: torch.tensor(
                    informativeness_label, dtype=torch.long
                ),
            },
            metadata={
                "caption": caption,
                "image_path": image_path,
                "topics": self.topics,
                "image_id": str(image_id)
            },
        )
        # self._validate_data_item(data_item, idx)
        return data_item

    def _validate_data_item(self, data_item: DataItemSchema, idx: int):
        try:
            assert data_item is not None, f"[IDX {idx}] DataItem is None!"
            assert isinstance(
                data_item.image_tensors, dict
            ), f"[IDX {idx}] image_tensors not dict"
            assert all(
                isinstance(v, ImageTensorsSchema)
                for v in data_item.image_tensors.values()
            ), f"[IDX {idx}] image_tensors values not ImageTensorsSchema"

            assert isinstance(
                data_item.tokenized_text_inputs, dict
            ), f"[IDX {idx}] tokenized_text_inputs not dict"
            for k, v in data_item.tokenized_text_inputs.items():
                assert isinstance(
                    v, TokenizedTextInputsSchema
                ), f"[IDX {idx}] tokenized_text_inputs[{k}] is not TokenizedTextInputsSchema"

            assert isinstance(
                data_item.topics_tokenized_inputs, list
            ), f"[IDX {idx}] topics_tokenized_inputs not list"
            for i, t in enumerate(data_item.topics_tokenized_inputs):
                assert isinstance(
                    t, dict
                ), f"[IDX {idx}] topics_tokenized_inputs[{i}] not dict"
                for key, val in t.items():
                    assert isinstance(
                        val, torch.Tensor
                    ), f"[IDX {idx}] topic_token[{key}] not tensor"

            assert isinstance(data_item.label, dict), f"[IDX {idx}] label not dict"
            for k, v in data_item.label.items():
                assert isinstance(v, torch.Tensor), f"[IDX {idx}] label[{k}] not tensor"

            print(f"[IDX {idx}] All fields in DataItemSchema are valid!")
        except AssertionError as e:
            print(f"[IDX {idx}] Validation error: {str(e)}")
            raise

    @classmethod
    def get_topics(cls) -> list[str]:
        config = config_utils.load_config()
        if config.get("train_bert_topic"):
            return []
        topics = []
        topic_model = get_tseqd_topic_model(None, None)
        num_classes = len(constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES)
        for topic in range(num_classes):
            curr_topics = topic_model.get_topic(topic, full=False)
            curr_topics = [{"topic": t[0], "score": t[1]} for t in curr_topics][
                : config.get(constants.FIELD_NUM_SELECTED_TOPICS) // num_classes
            ]
            topics.extend(curr_topics)
        assert len(topics) == config.get(constants.FIELD_NUM_SELECTED_TOPICS)
        return topics
