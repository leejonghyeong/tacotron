#tokenize
from tokenization_kobert import KoBertTokenizer
from .data.utils import kobert_tokenizer

def get_tokenized(infile, outfile):
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    with open(infile, "rt", encoding='UTF8') as f:
        text = f.readlines()
    kobert_tokenizer(text, outfile, tokenizer)

