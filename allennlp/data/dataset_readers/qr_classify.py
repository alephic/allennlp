
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField, LabelField
import json
from typing import Dict

@DatasetReader.register('qr_classify')
class QRClassifyDatasetReader(DatasetReader):
    def __init__(
            self, 
            token_indexers: Dict[str, TokenIndexer] = None,
            tokenizer: Tokenizer = None,
            lazy: bool=False
        ):
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer
        self._label_map = {'True': 1, 'Other': 1, 'False': 0, True: 1, False: 0}
    
    def text_to_instance(self, text, label=None):
        fields = {
            'tokens': TextField(self._tokenizer.tokenize(text), self._token_indexers)
        }
        if label is not None:
            fields['label'] = LabelField(self._label_map[label], skip_indexing=True)
        return Instance(fields)
    
    def _read(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                yield self.text_to_instance(obj['text'], label=obj.get('tag', None))
