
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from typing import Dict

@Model.register('logistic_regression')
class LogisticRegression(Model):
    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, num_labels: int):
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.linear = torch.nn.Linear(text_field_embedder.get_output_dim(), num_labels)
        self.acc = CategoricalAccuracy()
    
    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor = None):
        mask = get_text_field_mask(tokens)
        embedded = self.text_field_embedder(tokens)
        embedded = embedded.view(-1, embedded.size(-1))
        logits = self.linear(embedded)
        logits = logits.view(embedded.size(0), embedded.size(1), -1)
        logits = logits.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1).float()
        output_dict = {'logits': logits}
        if label is not None:
            output_dict['loss'] = torch.nn.functional.cross_entropy(logits, label)
            self.acc(logits, label)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.acc.get_metric(reset)}