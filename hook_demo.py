from transformers import BertForSequenceClassification, BertTokenizer


# First, define hooks functions for all possible hooks points in the model.

def embedding_output_hook(embedding_output):
    print("Hook at embedding output. Embedding output shape: ", embedding_output.shape)
    return embedding_output


def post_layer_hook(layer_id, hidden_states, attention_mask):
    print(f"Post-layer hook at layer {layer_id} with hidden states & attention mask of shapes: {hidden_states.shape} & {attention_mask.shape}")
    return hidden_states, attention_mask


def self_attn_ln_hook(hidden_states, input_tensor, layer_norm):
    print("Hook at self-attention layer with input hidden states of shape: ", hidden_states.shape)
    return layer_norm(hidden_states + input_tensor)


def final_ln_hook(hidden_states, input_tensor, layer_norm):
    print("Hook at output layer with input hidden states of shape: ", hidden_states.shape)
    return layer_norm(hidden_states + input_tensor)


# Load model and tokenizer as usual.

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Register all hooks by passing the function handles.
# Note that all methods will return PyTorch `RemovableHandle` objects that can be used to remove the hooks later.

handle1 = model.register_post_embedding_hook(embedding_output_hook)
handle2 = model.register_post_layer_hook(post_layer_hook)
handle3 = model.register_self_attn_ln_hook(11, self_attn_ln_hook)
handle4 = model.register_final_ln_hook(11, final_ln_hook)

# Try a forward pass and watch for the prints withing the hooks.

print("===== First run =====")
inputs = tokenizer("Hooks are great", return_tensors="pt")
outputs = model(**inputs)
print("Outputs: ", outputs[0])

# Remove some of the handles and try a new forward pass.

handle1.remove()
handle2.remove()

print("===== Second run =====")
outputs = model(**inputs)
print("Outputs: ", outputs[0])
