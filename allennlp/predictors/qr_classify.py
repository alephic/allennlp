from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('qr_classify')
class QRClassifyPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"text": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["text"]
        return self._dataset_reader.text_to_instance(sentence)