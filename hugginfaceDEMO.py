# from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
     
# from transformers import pipeline
# model_name2 = "nlptown/bert-base-multilingual-uncased-sentiment"
# mymodel2 = AutoModelForSequenceClassification.from_pretrained(model_name2)
# mytokenizer2 = AutoTokenizer.from_pretrained(model_name2)

# classifier = pipeline("sentiment-analysis", model = mymodel2 , tokenizer = mytokenizer2)
# res = classifier("I was so not happy with the Barbie Movie")
# print(res)

from transformers import AutoTokenizer
#load a pre trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#example text
text = "I was not happy with the barbie movie"

#Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens: ",tokens)

#convert tokens to input IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("input Ids: ", input_ids)

#encode the text( toeknization + converting to input IDs)
encoded_input = tokenizer(text)
print("Encoded input :",encoded_input)

#decode the text
decoded_output = tokenizer.decode(input_ids)
print("Decode Output", decoded_output)