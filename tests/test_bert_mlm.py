import torch
from transformers import BertTokenizer, BertForMaskedLM, pipeline

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

input_text = ("After Abraham Lincoln won the November 1860 presidential "
        "election on an anti-slavery platform, an initial seven "
        "slave states declared their secession from the country "
        "to form the Confederacy. War broke out in April 1861 "
        "when secessionist forces attacked Fort Sumter in South "
        "Carolina, just over a month after Lincoln's "
        "inauguration.")
        
tkn_input = tokenizer(input_text, return_tensors="pt")
tkn_input['labels'] = tkn_input.input_ids.detach().clone()

rand = torch.rand(tkn_input.input_ids.shape)
mask_arr = (rand < 0.15) * (tkn_input.input_ids != 101) * (tkn_input.input_ids != 102) 
_, mask_indexes = mask_arr.nonzero(as_tuple=True)
tkn_input.input_ids[0, mask_indexes] = 103

print()
print(f"masked token idx: {mask_indexes}")
print(tokenizer.convert_ids_to_tokens(tkn_input['input_ids'].tolist()[0]))

with torch.no_grad():
    output = model(**tkn_input)
    
    predicted_token_id = output.logits[0, mask_indexes].argmax(axis=-1)
    print(tokenizer.decode(predicted_token_id))

print(tokenizer.decode(tkn_input.input_ids[0], skip_special_tokens=True))
#unmasker = pipeline('fill-mask', model='bert-base-uncased')
#unmasker(input_text)




