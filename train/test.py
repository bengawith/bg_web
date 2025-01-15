from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
MODEL_PATH = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def chat_with_model(user_message):
    context = (
        "You are a helpful assistant answering questions about Benjamin Gawith. "
        "Only provide answers related to Benjamin's CV, skills, and projects.\n"
    )
    inputs = tokenizer.encode(context + f"User: {user_message}\nAssistant:", return_tensors="pt")
    reply_ids = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response



while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    print(f"Bot: {chat_with_model(user_input)}")