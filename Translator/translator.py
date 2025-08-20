from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

tokenizer.src_lang = "en_XX"

languages = [
    ("Hindi", "hi_IN"),
    ("Gujarati", "gu_IN"),
    ("Marathi", "mr_IN"),
    ("French", "fr_XX"),
    ("Arabic", "ar_AR"),
]

print("Choose a language to translate to:")
for i in range(len(languages)):
    print(f"{i + 1}. {languages[i][0]}")

text = input("\nEnter English text: ")

choice = int(input("Enter the number of the language: "))
target_lang = languages[choice - 1][1]

encoded_ar = tokenizer(text, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar,forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
translated = tokenizer.batch_decode(generated_tokens , skip_special_tokens=True)
print("\nTranslated text:")
print(translated)