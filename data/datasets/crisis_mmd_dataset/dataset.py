import torch
from typing import List, Dict
import pandas as pd
from transformers import RobertaTokenizer
from PIL import Image, ImageFile
from torchvision import transforms

from data.dataset_interface import DatasetInterface
from data.schemas import DataItemSchema, ImageTensorsSchema, TokenizedTextInputsSchema
from utils import config_utils, data_utils, caption_utils
import constants
from enums import SplitRunType
from topic_modelling.crisis_mmd import get_topic_model as get_crisis_mmd_topic_model

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_COL_TO_USE = "label"




class CrisisMMDDataset(DatasetInterface):
    def __init__(
        self,
        split_run_type: SplitRunType,
        tokenizer=caption_utils.get_default_tokenizer(),
    ) -> None:
        super().__init__(split_run_type)
        self._config = config_utils.load_config()

        assert self._config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_CRISIS_MMD_DATASET
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
        self.topics = CrisisMMDDataset.get_topics()

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
            for label in constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES
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

    def _load_dataframes_split(self) -> Dict[SplitRunType, pd.DataFrame]:
        train_df = pd.read_csv(
            f"{constants.PATH_CRISIS_MMD_DATASET_TSV_DIR}/task_humanitarian_text_img_agreed_lab_train.tsv",
            sep="\t",
        )
        validation_df = pd.read_csv(
            f"{constants.PATH_CRISIS_MMD_DATASET_TSV_DIR}/task_humanitarian_text_img_agreed_lab_dev.tsv",
            sep="\t",
        )
        test_df = pd.read_csv(
            f"{constants.PATH_CRISIS_MMD_DATASET_TSV_DIR}/task_humanitarian_text_img_agreed_lab_test.tsv",
            sep="\t",
        )

        train_df = train_df[
            train_df[LABEL_COL_TO_USE].isin(
                constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES
            )
        ]
        validation_df = validation_df[
            validation_df[LABEL_COL_TO_USE].isin(
                constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES
            )
        ]
        test_df = test_df[
            test_df[LABEL_COL_TO_USE].isin(
                constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES
            )
        ]

        # train_df = train_df[train_df["label_text_image"] == "Positive"]
        # validation_df = validation_df[validation_df["label_text_image"] == "Positive"]
        # test_df = test_df[test_df["label_text_image"] == "Positive"]

        # train_df = self.stratified_sample(train_df)
        # validation_df = self.stratified_sample(validation_df)
        # test_df = self.stratified_sample(test_df)

        assert sorted(list(set(train_df[LABEL_COL_TO_USE].tolist()))) == sorted(
            list(set(constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES))
        ), sorted(list(set(train_df[LABEL_COL_TO_USE].tolist())))

        assert sorted(list(set(validation_df[LABEL_COL_TO_USE].tolist()))) == sorted(
            list(set(constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES))
        ), sorted(list(set(validation_df[LABEL_COL_TO_USE].tolist())))

        assert sorted(list(set(test_df[LABEL_COL_TO_USE].tolist()))) == sorted(
            list(set(constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES))
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

        topics = CrisisMMDDataset.get_topics()

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
        image_dataset_id = self.dataframe.iloc[idx]["image_id"]
        event_name = self.dataframe.iloc[idx]["event_name"]

        relative_image_path = self.dataframe.iloc[idx]["image"]
        image_path = f"{constants.PATH_CRISIS_MMD_DATASET_BASE_PATH}/CrisisMMD_v2.0/{relative_image_path}"
        image = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)

        image_id = image_dataset_id + event_name + relative_image_path
        caption = self.dataframe.iloc[idx]["tweet_text"]
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

        humanitarian_category_label_name = self.dataframe.iloc[idx][LABEL_COL_TO_USE]
        humanitarian_category_label = (
            constants.LABEL_NAME_TO_LABEL_VALUE_CRISIS_MMD_HUMANITARIAN_CATEGORY[
                humanitarian_category_label_name
            ]
        )

        assert humanitarian_category_label >= 0 and humanitarian_category_label <= len(
            constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES
        )

        # Compute mean of non-NaN values
        nan_mask = torch.isnan(image_tensor)
        mean_value = torch.nanmean(image_tensor)  # Computes mean while ignoring NaNs

        # Replace NaNs with mean
        image_tensor[nan_mask] = mean_value

        data_item = DataItemSchema(
            image_tensors={
                constants.FIELD_HUMANITARIAN_CATEGORY: ImageTensorsSchema(
                    **{constants.FIELD_RGB_PIXELS_TENSOR: image_tensor}
                )
            },
            tokenized_text_inputs={
                constants.FIELD_HUMANITARIAN_CATEGORY: caption_text_tokenized_inputs
            },
            topics_tokenized_inputs=self.topics_tokenized_inputs,
            label={
                constants.FIELD_HUMANITARIAN_CATEGORY: torch.tensor(
                    humanitarian_category_label, dtype=torch.long
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
        topics = []
        config = config_utils.load_config()
        topic_model = get_crisis_mmd_topic_model(None, None)
        num_classes = len(constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES)
        for topic in range(num_classes):
            curr_topics = topic_model.get_topic(topic, full=False)
            curr_topics = [{"topic": t[0], "score": t[1]} for t in curr_topics][
                : config.get(constants.FIELD_NUM_SELECTED_TOPICS) // num_classes
            ]
            topics.extend(curr_topics)
        assert len(topics) == config.get(constants.FIELD_NUM_SELECTED_TOPICS)
        return topics
