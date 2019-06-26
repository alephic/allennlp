
{
    "dataset_reader": {
        "type": "qr_classify",
        "tokenizer": {
            "word_splitter": "bert-basic"
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "train_data_path": "/path/to/training/data",
    "validation_data_path": "/path/to/validation/data",
    "model": {
        "type": "logistic_regression",
        "num_labels": 2,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                    "embedding_dim": 300,
                    "trainable": false
                }
            }
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 10,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
