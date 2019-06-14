
local bert_model = "bert-base-uncased";

{
    "dataset_reader": {
        "type": "qr_classify",
        "tokenizer": {
            "word_splitter": "bert-basic"
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
                "max_pieces": 256
            }
        }
    },
    "train_data_path": "/path/to/training/data",
    "validation_data_path": "/path/to/validation/data",
    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "dropout": 0.2
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 5
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
