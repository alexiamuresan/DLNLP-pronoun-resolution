{
  "dataset_reader": {
    "type": "coref",
    "max_span_width": 30, 
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": "spanbert_local/",
        "max_length": 512
      }
    }
  },
  "train_data_path": "gap-development.jsonl",
  "validation_data_path":  "gap-validation.jsonl",
  "model": {
    "type": "coref",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": "SpanBERT/spanbert-coreference",
          "max_length": 512
        }
      }
    }
  },
  "data_loader": {
    "batch_size": 4
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "num_epochs": 3,
    "cuda_device": 0
  }
}
