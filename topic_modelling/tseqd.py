import sys 
sys.path.append('.')

import numpy as np
from enums import SplitRunType
from utils import config_utils, gpu_utils, caption_utils
from data.schemas import DataItemSchema, PytorchModelOutputSchema
import constants

from bertopic import BERTopic
from bertopic.representation import TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
import torch


config = config_utils.load_config()


TOPIC_MODEL_PATH = "./topic_modelling/tseqd_topic_model/"
LABEL_COL_TO_USE = "text_info_type"

class NeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size=config.get(constants.FIELD_EMBEDDING_DIM),
        hidden_size=config.get(constants.FIELD_EMBEDDING_DIM),
        output_size=len(constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES),
    ):
        super(NeuralNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, dtype=torch.float64),
        )

    def forward(self, x):
        return self.model(x)


class CustomClassifierBERTopic:
    def __init__(self):
        self.model = NeuralNetwork().to(gpu_utils.get_device())
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, X, y, epochs=100, batch_size=32):
        X = torch.from_numpy(X).to(torch.float64)
        y = torch.from_numpy(y).to(torch.long)
        X = X.to(gpu_utils.get_device())
        y = y.to(gpu_utils.get_device())

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        try:
            X = torch.from_numpy(X).to(torch.float64)
        except:
            X = torch.tensor(X).to(torch.float64)
        X = X.to(gpu_utils.get_device())
        outputs = self.model(X)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy().tolist()

def train_bert_topic(docs, y):
    empty_dimensionality_model = BaseDimensionalityReduction()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    topic_model = BERTopic(
        embedding_model="distilroberta-base-msmarco-v1",
        umap_model=empty_dimensionality_model,
        hdbscan_model=CustomClassifierBERTopic(),
        ctfidf_model=ctfidf_model,
        top_n_words=15
    ) 
    topic_model.fit_transform(docs, y=y)
    topic_model.save(
        TOPIC_MODEL_PATH,
        serialization="pytorch",
        save_ctfidf=True,
        save_embedding_model=empty_dimensionality_model,
    )
    print()
    for topic in range(len(constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES)):
        topics = topic_model.get_topic(topic, full=False)
        print(topics)
    print()
    return topic_model

def get_topic_model(docs, y):
    if config.get("train_bert_topic"):
        print('----- TRAININING TOPIC MODEL -----')
        return train_bert_topic(docs, y)
    else:
        print('----- USING TRAINED TOPIC MODEL -----')
        return BERTopic.load(TOPIC_MODEL_PATH)
    
if __name__ == "__main__":
    from data.datasets import TSEQDDataset
    dataset = TSEQDDataset(split_run_type=SplitRunType.TRAIN)
    train_df = dataset._get_dataframe_by_split_run_type()

    texts =  train_df["text"].tolist()
    informative_category_labels = np.array(train_df[LABEL_COL_TO_USE])
    texts = [caption_utils.clean_text(t) for t in texts]

    print("Label range:", informative_category_labels.min(), informative_category_labels.max())
    print("Unique labels:", set(informative_category_labels))

    topic_model = get_topic_model(docs=texts, y=informative_category_labels)