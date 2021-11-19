#tokenize
from tokenization_kobert import KoBertTokenizer
import json

def kobert_tokenizer(text, outfile, tokenizer : KoBertTokenizer):
    '''
    text: list of the following sentences. 
    e.g.
        1/1_0002.wav|용돈을 아껴 써라.|용돈을 아껴 써라.|용돈을 아껴 써라.|1.8|Save your pocket money.
    '''
    with open(outfile, "wt", encoding='UTF8') as f:
        for line in text:
            sentence = line.split('|')[1]
            token = tokenizer.tokenize(sentence)
            token_ids = tokenizer.convert_tokens_to_ids(token)

            f.write(json.dumps(token_ids+"\n"))

def get_tokenized(infile, outfile):
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    with open(infile, "rt", encoding='UTF8') as f:
        text = f.readlines()
    kobert_tokenizer(text, outfile, tokenizer)

