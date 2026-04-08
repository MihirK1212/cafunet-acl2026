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

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

config = config_utils.load_config()


class MedicDisasterDataset(DatasetInterface):
    def __init__(
        self,
        split_run_type: SplitRunType,
        tokenizer=caption_utils.get_default_tokenizer(),
    ) -> None:
        super().__init__(split_run_type)
        self.dataframes_split = self._load_dataframes_split()
        self.dataframe = self._load_dataframe()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the image
                transforms.ToTensor(),
            ]
        )
        self.tokenizer = tokenizer
        self.analyse_dataframe()

    def analyse_dataframe(self):
        print()
        print(self.split_run_type)
        print("size:", self.dataframe.shape)
        print(self.dataframe.head())
        print()

    def stratified_sample(self, df, sample_fraction=0.4):
        # Get rows for each label in class_a
        class_a_rows = [
            df[df["disaster_types"] == label].sample(n=1, random_state=42)
            for label in constants.LABELS_MEDIC_DISASTER_TYPES
            if label in df["disaster_types"].values
        ]
        class_a_rows_df = pd.concat(class_a_rows).drop_duplicates()

        # Get rows for each label in class_b
        class_b_rows = [
            df[df["humanitarian"] == label].sample(n=1, random_state=42)
            for label in constants.LABELS_MEDIC_HUMANITARIAN_CATEGORIES
            if label in df["humanitarian"].values
        ]
        class_b_rows_df = pd.concat(class_b_rows).drop_duplicates()

        # Combine rows from both classes
        label_rows_df = pd.concat([class_a_rows_df, class_b_rows_df]).drop_duplicates()

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
            f"{constants.PATH_MEDIC_DISASTER_DATASET_TSV_DIR}/MEDIC_train.tsv", sep="\t"
        )
        validation_df = pd.read_csv(
            f"{constants.PATH_MEDIC_DISASTER_DATASET_TSV_DIR}/MEDIC_dev.tsv", sep="\t"
        )
        test_df = pd.read_csv(
            f"{constants.PATH_MEDIC_DISASTER_DATASET_TSV_DIR}/MEDIC_test.tsv", sep="\t"
        )

        # train_df = self.stratified_sample(train_df)
        # validation_df = self.stratified_sample(validation_df)
        # test_df = self.stratified_sample(test_df)

        assert sorted(list(set(train_df["disaster_types"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_DISASTER_TYPES))
        )
        assert sorted(list(set(validation_df["disaster_types"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_DISASTER_TYPES))
        )
        assert sorted(list(set(test_df["disaster_types"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_DISASTER_TYPES))
        )

        assert sorted(list(set(train_df["humanitarian"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_HUMANITARIAN_CATEGORIES))
        )
        assert sorted(list(set(validation_df["humanitarian"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_HUMANITARIAN_CATEGORIES))
        )
        assert sorted(list(set(test_df["humanitarian"].tolist()))) == sorted(
            list(set(constants.LABELS_MEDIC_HUMANITARIAN_CATEGORIES))
        )

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

    def get_data_item(self, idx=-1) -> DataItemSchema:
        image_dataset_id = self.dataframe.iloc[idx]["image_id"]
        event_name = self.dataframe.iloc[idx]["event_name"]

        relative_image_path = self.dataframe.iloc[idx]["image_path"]
        image_path = (
            f"{constants.PATH_MEDIC_DISASTER_DATASET_BASE_PATH}/{relative_image_path}"
        )
        image = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)

        image_id = image_dataset_id + event_name + relative_image_path
        caption = caption_utils.get_caption(
            image_id=image_id, image_file_path=image_path
        )
        caption_text_tokenized_inputs = data_utils.parse_tokenizer_output(
            self.tokenizer(
                caption,
                add_special_tokens=True,
                max_length=config.get(constants.FIELD_MAX_LEN_TEXT),
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )
        )

        disaster_type_label_name = self.dataframe.iloc[idx]["disaster_types"]
        disaster_type_label = constants.LABEL_NAME_TO_LABEL_VALUE_MEDIC_DISASTER_TYPES[
            disaster_type_label_name
        ]

        humanitarian_category_label_name = self.dataframe.iloc[idx]["humanitarian"]
        humanitarian_category_label = (
            constants.LABEL_NAME_TO_LABEL_VALUE_MEDIC_HUMANITARIAN_CATEGORY[
                humanitarian_category_label_name
            ]
        )

        if caption is None:
            caption = ""

        return DataItemSchema(
            image_tensors={
                constants.FIELD_DISASTER_TYPE: ImageTensorsSchema(
                    **{constants.FIELD_RGB_PIXELS_TENSOR: image_tensor}
                )
            },
            tokenized_text_inputs={
                constants.FIELD_DISASTER_TYPE: caption_text_tokenized_inputs
            },
            label={
                constants.FIELD_DISASTER_TYPE: torch.tensor(
                    disaster_type_label, dtype=torch.long
                ),
                constants.FIELD_HUMANITARIAN_CATEGORY: torch.tensor(
                    humanitarian_category_label, dtype=torch.long
                ),
            },
            metadata={
                constants.FIELD_IMAGE_ID: str(image_id),
                'clip_text': " ".join(caption.split()[:30]),
            },
        )
