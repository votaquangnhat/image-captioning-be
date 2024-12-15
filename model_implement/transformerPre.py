from transformers import VisionEncoderDecoderModel

def TransformerPre(tokenizer):
    # Load models and tokenizer
    encoder_model = "google/vit-base-patch16-224-in21k"
    decoder_model = "bert-base-uncased"

    # Initialize EncoderDecoderModel
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model, decoder_model
    )

    # Set decoder config to force the model to attend to encoder outputs
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 50   # Maximum token length for generated captions
    model.config.min_length = 5    # Minimum token length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.config.attention_dropout = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.config.gradient_checkpointing = True

    return model