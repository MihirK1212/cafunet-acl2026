from model_interfaces import ModelInterface
from data.schemas import DataItemSchema

class ClassicalModelInteface(ModelInterface):
    def __init__(self, model):
        super().__init__(model)

    def _get_model_output(self, data_item: DataItemSchema):
        return self.model.predict(data_item)
    
    def _get_prediction_from_model_output(self, model_output):
        return model_output['predictions']
    
    def fit(self, data_item: DataItemSchema):
        self.model.train(data_item)