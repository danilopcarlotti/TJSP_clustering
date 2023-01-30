from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import sys
sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent) + "/Classic_ml/")

import assets
from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex

class SentenceEmbedder:
    
    #https://huggingface.co/rufimelo/Legal-BERTimbau-sts-base-ma-v2
    #tokenizer = AutoTokenizer.from_pretrained('rufimelo/Legal-BERTimbau-sts-base-ma-v2', model_max_length=512)
    #model = AutoModel.from_pretrained('rufimelo/Legal-BERTimbau-sts-base-ma-v2')
    
    tokenizer = AutoTokenizer.from_pretrained('rufimelo/Legal-BERTimbau-sts-base-ma-v2', model_max_length=512)
    model = AutoModel.from_pretrained('ulysses-camara/legal-bert-pt-br/O_Transformer')
    
    def __init__(self):
        print(self.model.config)
        
    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def eval(self, input_sentences):
        tkn_input = self.tokenizer(input_sentences, padding=True, truncation=True, return_tensors='pt')  
        
        if tkn_input.input_ids.shape[1] > 512:
            print(f"TRUNCATION WARNING !: tokenized len = {tkn_input.input_ids.shape[1]}(>512)")      
        
        with torch.no_grad():
            output = self.model(**tkn_input)
            sentence_embeddings = self._mean_pooling(output, tkn_input['attention_mask'])
            
            return sentence_embeddings

if __name__ == "__main__":       
    
    data = assets.DataLoader(theme_id="S0929")
    se = SentenceEmbedder()

    for _, (did, theme, text) in zip(range(1), data):
        
        #TODO: implementar uma funcao que devolva uma lista de frases e nao um unico paragrafo
        dic_data_text = classifier_legal_sections_regex(text)
    
        print(f"\033[41mid:\033[0m {did}\n\033[41mtheme:\033[0m {theme}\n")
        
        for k, v in dic_data_text.items():
            print(f"\033[41m{k}:\033[0m \"{v}\" \n")
        
        #pode-se incluir os demais segmentos do texto
        sentences = [dic_data_text["fato"]]
        
        #recebe uma lista com as frases que serao codificadas.
        embeddings = se.eval(sentences)
        
        torch.set_printoptions(threshold=5)
        for s,e in zip(sentences, embeddings):
            print(f"\n\033[42mSentence:\033[0m {s}")
            print(f"\033[42mEmbeddings:\033[0m length ({e.size()}) {e}")

