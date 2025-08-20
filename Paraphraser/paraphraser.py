from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

def paraphrase_text(text):
    
    input_text = f"paraphrase: {text} </s>"
    encoding = tokenizer.encode_plus(
        input_text,
        padding='max_length',
        max_length=256,
        return_tensors="pt"
        
    )

    outputs = model.generate(
    input_ids=encoding["input_ids"],
    attention_mask=encoding["attention_mask"],
    temperature=1.0,
    top_k=50,                    
    top_p=0.95,
    max_length=256,
    do_sample=True,                           
    num_return_sequences=5,

    )
    paraphrases = [
        tokenizer.decode(output) 
         for output in outputs
    ]
    return paraphrases

if __name__ == "__main__":
    user_input = input("Enter a sentence:").strip()
    results = paraphrase_text(user_input, num_return_sequences=5)

    print("\nParaphrasing:\n")
    for i, para in enumerate(results, 1):
        print(f"{i}. {para}")
