FIELD_IMAGE_TENSORS = "image_tensors"
FIELD_TOKENIZED_TEXT_INPUTS = "tokenized_text_inputs"
FIELD_LABEL = "label"
FIELD_METADATA = "metadata"
FIELD_SEED = "seed"
FIELD_TRAIN_SPLIT_RATIO = "train_split_ratio"
FIELD_MAX_LEN_TEXT = "max_len_text"
FIELD_TRAIN_BATCH_SIZE = "train_batch_size"
FIELD_VALIDATION_BATCH_SIZE = "validation_batch_size"
FIELD_TEST_BATCH_SIZE = "test_batch_size"
FIELD_DISASTER_TYPE = "disaster_type"
FIELD_HUMANITARIAN_CATEGORY = "humanitarian"
FIELD_INFORMATIVENESS_CATEGORY = "informativeness"
FIELD_IMAGE_ID = "image_id"
FIELD_RGB_PIXELS_TENSOR = "rgb_pixels_tensor"
FIELD_PRED_LOGITS = "pred_logits"
FIELD_PYTROCH_MODEL_INTERFACE = "pytorch_model_interface"
FIELD_CLASSICAL_MODEL_INTERFACE = "classical_model_interface"
FIELD_APPLY_GRADIENT_CLIPPING = "apply_gradient_clipping"
FIELD_USE_LR_SCHEDULER = "use_lr_scheduler"
FIELD_DATASET = "dataset"
FIELD_MODEL_INTERFACE = "model_interface"
FIELD_RUNNER_INTERFACE = "runner_interface"
FIELD_MODEL_TO_USE = "model_to_use"
FIELD_EMBEDDING_DIM = "embedding_dim"
FIELD_IMAGE_EMBEDDING_DIM = "image_embedding_dim"
FIELD_TEXT_EMBEDDING_DIM = "text_embedding_dim"
FIELD_GLOBAL_EMBEDDING = "global_embedding"
FIELD_WORD_LEVEL_EMBEDDINGS = "word_level_embeddings"
FIELD_METADATA = "metadata"
FIELD_TEXT_EMBEDDINGS = "text_embeddings"
FIELD_IMAGE_EMBEDDINGS = "image_embeddings"
FIELD_CL_LOSS_TEMPERATURE = 'cl_loss_temperature'
FIELD_TEXT = "text"
FIELD_NUM_SELECTED_TOPICS = "num_selected_topics"
FIELD_USE_DUMMY_RANK_FOR_PARALLEL_ENABLED_MODEL = "use_dummy_rank_for_parallel_enabled_model"
FIELD_RUN_PARALLEL = "run_parallel"
FIELD_NUM_EPOCHS = "num_epochs"
FIELD_READ_WRITE_GPU_RANK = "read_write_gpu_rank"
FIELD_VOCAB_SIZE = "vocab_size"
FIELD_TRANSFORMER_TEXT_CONTEXT_LENGTH = "transformer_text_context_length"


PATH_CONFIG_FILE = "config.yaml"

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"

############################### MEDIC DISASTER DATASET ###############################
PATH_MEDIC_DISASTER_DATASET_BASE_PATH = (
    "./multimodal-disaster-datasets/medic-crisis-nlp"
)
PATH_MEDIC_DISASTER_DATASET_TSV_DIR = (
    PATH_MEDIC_DISASTER_DATASET_BASE_PATH + "/" + "tsvs"
)

LABEL_MEDIC_EARTHQUAKE = "earthquake"
LABEL_MEDIC_NOT_DISASTER = "not_disaster"
LABEL_MEDIC_FLOOD = "flood"
LABEL_MEDIC_LANDSLIDE = "landslide"
LABEL_MEDIC_FIRE = "fire"
LABEL_MEDIC_HURRICANE = "hurricane"
LABEL_MEDIC_OTHER_DISASTER = "other_disaster"

LABEL_MEDIC_INFRASTRUCTURE_AND_UTILITY_DAMAGE = "infrastructure_and_utility_damage"
LABEL_MEDIC_AFFECTED_INJURED_DEAD_PEOPLE = "affected_injured_or_dead_people"
LABEL_MEDIC_RECUE_VOLUNTEERING_DONATION_EFFORT = (
    "rescue_volunteering_or_donation_effort"
)
LABEL_MEDIC_NOT_HUMANITARIAN = "not_humanitarian"

LABELS_MEDIC_DISASTER_TYPES = [
    LABEL_MEDIC_EARTHQUAKE,
    LABEL_MEDIC_NOT_DISASTER,
    LABEL_MEDIC_FLOOD,
    LABEL_MEDIC_LANDSLIDE,
    LABEL_MEDIC_FIRE,
    LABEL_MEDIC_HURRICANE,
    LABEL_MEDIC_OTHER_DISASTER,
]

LABELS_MEDIC_HUMANITARIAN_CATEGORIES = [
    LABEL_MEDIC_INFRASTRUCTURE_AND_UTILITY_DAMAGE,
    LABEL_MEDIC_AFFECTED_INJURED_DEAD_PEOPLE,
    LABEL_MEDIC_RECUE_VOLUNTEERING_DONATION_EFFORT,
    LABEL_MEDIC_NOT_HUMANITARIAN,
]

LABEL_NAME_TO_LABEL_VALUE_MEDIC_DISASTER_TYPES = {
    LABEL_MEDIC_EARTHQUAKE: 0,
    LABEL_MEDIC_FLOOD: 1,
    LABEL_MEDIC_LANDSLIDE: 2,
    LABEL_MEDIC_FIRE: 3,
    LABEL_MEDIC_HURRICANE: 4,
    LABEL_MEDIC_OTHER_DISASTER: 5,
    LABEL_MEDIC_NOT_DISASTER: 6,
}

LABEL_NAME_TO_LABEL_VALUE_MEDIC_HUMANITARIAN_CATEGORY = {
    LABEL_MEDIC_INFRASTRUCTURE_AND_UTILITY_DAMAGE: 0,
    LABEL_MEDIC_AFFECTED_INJURED_DEAD_PEOPLE: 1,
    LABEL_MEDIC_RECUE_VOLUNTEERING_DONATION_EFFORT: 2,
    LABEL_MEDIC_NOT_HUMANITARIAN: 3,
}

NUM_LABELS_MEDIC = {}
####################################################################################


############################### CRISIS MMD DATASET ###############################
PATH_CRISIS_MMD_DATASET_BASE_PATH = (
    "../Multimodal-Disaster-Classification/multimodal-disaster-datasets/crisis-mmd"
)
PATH_CRISIS_MMD_DATASET_TSV_DIR = (
    PATH_CRISIS_MMD_DATASET_BASE_PATH + "/" + "crisismmd_datasplit_agreed_label"
)

LABEL_CRISIS_MMD_NOT_HUMANITARIAN = "not_humanitarian"
LABEL_CRISIS_MMD_OTHER_RELEVAN_INFORMATION = "other_relevant_information"
LABEL_CRISIS_MMD_INFRASTRUCTURE_AND_UTILITY_DAMAGE = "infrastructure_and_utility_damage"
# LABEL_CRISIS_MMD_VEHICLE_DAMANGE = "vehicle_damage"
# LABEL_CRISIS_MMD_AFFECTED_INDIVIDUALS = "affected_individuals"
LABEL_CRISIS_MMD_RECUE_VOLUNTEERING_DONATION_EFFORT = (
    "rescue_volunteering_or_donation_effort"
)
# LABEL_CRISIS_MMD_INJURED_DEAD_PEOPLE = "injured_or_dead_people"
# LABEL_CRISIS_MMD_MISSING_FOUND_PEOPLE = "missing_or_found_people"

LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES = [
    LABEL_CRISIS_MMD_NOT_HUMANITARIAN,
    LABEL_CRISIS_MMD_OTHER_RELEVAN_INFORMATION,
    LABEL_CRISIS_MMD_INFRASTRUCTURE_AND_UTILITY_DAMAGE,
    # LABEL_CRISIS_MMD_VEHICLE_DAMANGE,
    # LABEL_CRISIS_MMD_AFFECTED_INDIVIDUALS,
    LABEL_CRISIS_MMD_RECUE_VOLUNTEERING_DONATION_EFFORT,
    # LABEL_CRISIS_MMD_INJURED_DEAD_PEOPLE,
    # LABEL_CRISIS_MMD_MISSING_FOUND_PEOPLE
]   

LABEL_NAME_TO_LABEL_VALUE_CRISIS_MMD_HUMANITARIAN_CATEGORY = {
    LABEL_CRISIS_MMD_NOT_HUMANITARIAN: 0,
    LABEL_CRISIS_MMD_OTHER_RELEVAN_INFORMATION: 1,
    LABEL_CRISIS_MMD_INFRASTRUCTURE_AND_UTILITY_DAMAGE: 2,
    # LABEL_CRISIS_MMD_VEHICLE_DAMANGE: 3,
    # LABEL_CRISIS_MMD_AFFECTED_INDIVIDUALS: 4,
    LABEL_CRISIS_MMD_RECUE_VOLUNTEERING_DONATION_EFFORT: 3,
    # LABEL_CRISIS_MMD_INJURED_DEAD_PEOPLE: 6,
    # LABEL_CRISIS_MMD_MISSING_FOUND_PEOPLE: 7
}
####################################################################################


############################### TSEQD DATASET ###############################
PATH_TSEQD_DATASET_BASE_PATH = (
    "./multimodal-disaster-datasets/tseqd"
)
PATH_TSEQD_DATASET_TSV_PATH = (
    PATH_TSEQD_DATASET_BASE_PATH + "/" + "updated_TSEQD_datasetfile.tsv"
)

LABELS_TSEQD_INFORMATIVENESS_CATEGORIES = [
    0,
    1,
    2,
    3,
    # 4,
    # 5,
    # 6,
    # 7
] 
####################################################################################

############################### NEPAL TWEETS DATASET ###############################
PATH_NEPAL_QUAKE_TWEETS_DATASET = (
    "./multimodal-disaster-datasets/nepal-quake-tweets/nepal-quake-dataset.csv"
)
ID_COLUMN = "tweet_id"
TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "tweet_label"
LABEL_NAME_COLUMN = "tweet_label_name"
####################################################################################

################################ DATASET KEY NAMES ################################
FIELD_CRISIS_MMD_DATASET = "crisis_mmd_dataset"
FIELD_TSEQD_DATASET = "tseqd_dataset"
####################################################################################

############################### MODEL_NAMES ###############################
MODEL_MEDIC_RESNET_BASIC = "model_medic_resnet_basic"
MODEL_MEDIC_VIT_BASIC = "model_medic_vit_basic"
MODEL_MEDIC_DUMMY_CAPTION_GENERATION = "model_medic_dummy_caption_generation"
MODEL_MEDIC_VISION_TEXT_PIPELINE = "model_medic_vision_text_pipeline"
MODEL_MEDIC_CLIP = "model_medic_clip"
MODEL_MEDIC_BERT = "model_medic_bert"

MODEL_CRISIS_MMD_RESNET_BASIC = "model_crisis_mmd_resnet_basic"
MODEL_CRISIS_MMD_VIT_BASIC = "model_crisis_mmd_vit_basic"
MODEL_CRISIS_MMD_VISION_TEXT_PIPELINE = "model_crisis_mmd_vision_text_pipeline"
MODEL_CRISIS_MMD_CLIP_BASIC = "model_crisis_mmd_clip_basic"
MODEL_CRISIS_MMD_FUZZY_ENSEMBLE = "model_crisis_mmd_fuzzy_ensemble"

MODEL_SOTA_1_SPRINGER_MULTIMODAL_ASSOCIATION = "model_sota_1_springer_multimodal_association"
MODEL_SOTA_2_ABIVASNI_DENSENET = "model_sota_2_abivasni_densenet"
MODEL_SOTA_3_DMCC = "model_sota_3_dmcc"
MODEL_SOTA_4_ARNAV_P1 = "sota_4_arnav_p1"
MODEL_SOTA_5_ARNAV_P2 = "sota_5_arnav_p2"
MODEL_SOTA_6_ARNAV_P3 = "sota_6_arnav_p3"
MODEL_SOTA_7_ARNAV_P4 = "sota_7_arnav_p4"
MODEL_SOTA_9_ARNAV_9_MMDF = "sota_9_arnav_9_mmdf"
MODEL_SOTA_10_ARNAV_10 = "sota_10_arnav_10"
MODEL_SOTA_11_ARNAV_11 = "sota_11_arnav_11"


MODEL_SAMPLE_LLM = "model_sample_llm"
MODEL_MISTRAL_LLM = "model_mistral_llm"
MODEL_GEMINI_LLM = "model_gemini_llm"
MODEL_LLAMA_LLM = "model_llama_llm"
####################################################################################
