import os
import torch
from transformers import BertTokenizer, BertForMaskedLM, pipeline
from torchtext.datasets import AG_NEWS

torch.hub.set_dir("/home/padilha/DATA2/pytorch/hub/")
print(torch.hub.get_dir())
os.environ['TRANSFORMERS_CACHE'] = '/home/padilha/DATA2/pytorch/hub/'
os.environ['HF_HOME'] = '/home/padilha/DATA2/pytorch/hub/'

#unmasker = pipeline('fill-mask', model='bert-base-uncased')
#unmasker(input_text)

input_text = (" Trata-se de recurso de apelação interposto " 
              "contra a r. decisão de fls  60/62, cujo relatório "
              "se adota, que julgou improcedente o pedido inicial, "
              "condenando o autor ao pagamento das custas, despesas "
              "processuais e honorários advocatícios, arbitrados em R$ 200,00. "
              "Aduziu, em suma, que se viu obrigado a pagar tarifas abusivas "
              "e indevidas, que resultaram na cobrança a maior de mais de "
              "R$ 2.000,00, quantia que deve ser restituída, acolhendo-se o "
              "pedido inicial. Sustentou que a cobrança indevida foi "
              "realizada de má-fé, devendo haver a devolução em dobro "
              "argumentando, no mais, pela reforma da r. decisão, dando-se "
              "provimento ao recurso interposto.")       

class BERT_MLM_eval:
    
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', cache_dir='/home/padilha/DATA2/pytorch/hub/')
    model = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased', cache_dir='/home/padilha/DATA2/pytorch/hub/')
    
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


