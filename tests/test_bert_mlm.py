import torch
from transformers import BertTokenizer, BertForMaskedLM, pipeline
from torchtext.datasets import AG_NEWS

#unmasker = pipeline('fill-mask', model='bert-base-uncased')
#unmasker(input_text)

input_text = ("After Abraham Lincoln won the November 1860 presidential "
        "election on an anti-slavery platform, an initial seven "
        "slave states declared their secession from the country "
        "to form the Confederacy. War broke out in April 1861 "
        "when secessionist forces attacked Fort Sumter in South "
        "Carolina, just over a month after Lincoln's "
        "inauguration.")       

class BERT_MLM_eval:
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    def __init__(self):
        pass
    
    def eval_15(self, input_text):
        
        tkn_input = self.tokenizer(input_text, return_tensors="pt")
        tkn_input['labels'] = tkn_input.input_ids.detach().clone()

        rand = torch.rand(tkn_input.input_ids.shape)
        mask_arr = (rand < 0.15) * (tkn_input.input_ids != 101) * (tkn_input.input_ids != 102) 
        _, mask_indexes = mask_arr.nonzero(as_tuple=True)
        original_masked_tkn = self.tokenizer.convert_ids_to_tokens(tkn_input.input_ids[0, mask_indexes])
        tkn_input.input_ids[0, mask_indexes] = 103

        print(f"\n\033[92mOriginal text:\033[0m\n {input_text}\n")
        #print()
        #print(f"masked token idx: {mask_indexes}")
        print(f"\n\033[92mMasked input:\033[0m\n{self.tokenizer.convert_ids_to_tokens(tkn_input['input_ids'].tolist()[0])}\n")
        print(f"\n\033[92mOriginal masked tkn:\033[0m\n{original_masked_tkn}\n")
        with torch.no_grad():
            output = self.model(**tkn_input)
            
            predicted_token_ids = output.logits[0, mask_indexes].argmax(axis=-1)
            print(f"\033[92mPredicted tokens:\033[0m\n{self.tokenizer.convert_ids_to_tokens(predicted_token_ids)}\n")

        tkn_input.input_ids[0, mask_indexes] = predicted_token_ids
        print(f"\033[92mFinal output:\033[0m\n\"{self.tokenizer.decode(tkn_input.input_ids[0], skip_special_tokens=True)}\"")
        

e = BERT_MLM_eval()
e.eval_15(input_text)
#for c, t in AG_NEWS(split='test'):
#    e.eval_15(t)


