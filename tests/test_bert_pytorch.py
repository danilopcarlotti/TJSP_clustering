import torch
import sys

torch.hub.set_dir("/home/padilha/DATA2/pytorch/hub/")
print(torch.hub.get_dir())

# Using the configuration with a model
config = torch.hub.load(
    "huggingface/pytorch-transformers", "config", "bert-base-uncased"
)
config.output_attentions = True
config.output_hidden_states = True
model = torch.hub.load(
    "huggingface/pytorch-transformers", "model", "bert-base-uncased", config=config
)
# Model will now output attentions and hidden states as well

tokenizer = torch.hub.load(
    "huggingface/pytorch-transformers", "tokenizer", "bert-base-uncased"
)

text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

print()
print(indexed_tokens)
print(tokenizer.convert_ids_to_tokens(indexed_tokens))

# sys.exit(0)

# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids])

masked_index = 11
indexed_tokens[masked_index] = tokenizer.mask_token_id
print(tokenizer.convert_ids_to_tokens(indexed_tokens))
tokens_tensor = torch.tensor([indexed_tokens])

masked_lm_model = torch.hub.load(
    "huggingface/pytorch-transformers", "modelForMaskedLM", "bert-base-uncased"
)

with torch.no_grad():
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)

# Get the predicted token
predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(predicted_token)
assert predicted_token == "Jimmy"
