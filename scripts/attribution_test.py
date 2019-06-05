
from allennlp.predictors.bert_attribution import BertMCAttributionPredictor
from allennlp.data.dataset_readers.bert_mc_qa import BertMCQAReader
import allennlp.models.bert_models
from allennlp.models.archival import load_archive
import sys
import logging
import json
import torch

def make_predictor(model_archive_path, cuda_device=-1, kwargs):
    a = load_archive(model_archive_path, cuda_device=cuda_device)
    reader_conf = a.config['dataset_reader']
    reader_conf.pop('type')
    reader = BertMCQAReader.from_params(reader_conf)
    return BertMCAttributionPredictor(a.model, reader, **kwargs)

def test(model_archive_path, data_path, predictor_kwargs):
    logging.basicConfig(level=logging.WARNING)
    predictor = make_predictor(model_archive_path, cuda_device=0 if torch.cuda.is_available() else -1, predictor_kwargs)
    with open(data_path) as f:
        data_lines = f.readlines()
    example = json.loads(data_lines[0])
    predictor.predict_json(example)

if __name__ == "__main__":
    predictor_kwarg_types = {'baseline_type': str, 'grad_sample_count': int}
    kwargs = {}
    arg_name = None
    for arg in sys.argv[3:]:
        if arg.startswith('--'):
            arg_name = arg[2:]
            if '=' in arg_name:
                arg_name, arg = arg_name.split('=')
                if arg_name in predictor_kwarg_types:
                    kwargs[arg_name] = predictor_kwarg_types[arg_name][arg]
                else:
                    raise RuntimeError(f"Unexpected argument \"{arg_name}\"")
                arg_name = None
        else:
            if arg_name is not None:
                if arg_name in predictor_kwarg_types:
                    kwargs[arg_name] = predictor_kwarg_types[arg_name][arg]
                else:
                    raise RuntimeError(f"Unexpected argument \"{arg_name}\"")
            arg_name = None
    test(sys.argv[1], sys.argv[2], kwargs)
