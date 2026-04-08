import json
import os
from transformers import RobertaTokenizer

from caption_generation.llava_captions import LlavaCaptionGeneration
from utils import config_utils, data_utils
import constants

config = config_utils.load_config()

def get_caption_from_image_file_path(image_file_path):
    return LlavaCaptionGeneration(image_file_path).get_caption()


def get_caption_from_json_data(
    id, json_file_path, image_file_path, update_forcefully=False
):
    #TODO optimize this dont load file everytime 
    
    # Check if the JSON file exists
    if os.path.exists(json_file_path) and (not update_forcefully):
        # Read the JSON file as a dictionary
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        # Check if the ID is present in the data
        if id in data:
            return data[id]["caption"]

    # ID not found or file doesn't exist, get the value from some_func
    value = get_caption_from_image_file_path(image_file_path)

    # Update the data with the new ID and value
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
    else:
        data = {}

    data[id] = {"caption": value, "generation_time_stamp": data_utils.get_timestamp()}

    # Write the updated data back to the JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return value


def get_caption(image_id, image_file_path, update_forcefully=False):
    assert 'medic' in config.get(constants.FIELD_MODEL_TO_USE)
    captions_json_path = f"./caption_generation/captions/{constants.MODEL_MEDIC_DUMMY_CAPTION_GENERATION}.json"
    return get_caption_from_json_data(
        id=image_id,
        json_file_path=captions_json_path,
        image_file_path=image_file_path,
        update_forcefully=update_forcefully,
    )

def get_default_tokenizer():
    return RobertaTokenizer.from_pretrained(
        "roberta-base", truncation=True, do_lower_case=True
    )


from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')


def clean_text(text):
    import re
    import string
    
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9_]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-Z0-9.,!?']", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    words = text.split()
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in ['rt', 'st', 'gt']]
    
    return ' '.join(words)

def pad_caption(caption: str, max_len_text: int) -> str:
    words = caption.split()
    padded_words = words + ["<PAD>"] * (max_len_text - len(words))
    return " ".join(padded_words)