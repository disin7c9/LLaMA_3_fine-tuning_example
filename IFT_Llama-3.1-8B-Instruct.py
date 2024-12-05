import torch
from transformers import (
    AutoTokenizer,
    TextStreamer,
)
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    unsloth_train,
    UnslothTrainer, 
    UnslothTrainingArguments,
)
from datasets import load_dataset

# Hugging Face repository settings
HF_write_token = "your_token"
user_name = "your_id"

seed = 42

# Important hyperparameters
max_seq_length = 2048
load_in_4bit = True
BATCH_SIZE = 16
rank = 128
alpha = rank*2

# repo path
base_model_path = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
repo_name = "CPT_LoRA_Llama-3.1-8B-Instruct-bnb-4bit_wikipedia-ko"
CPT_path = f"{user_name}/{repo_name}"


'''Instruction Fine-tuning'''

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CPT_path,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=load_in_4bit,
)

model.print_trainable_parameters()


# dataset formatting function
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(conversations):
    alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

    ### 지침:
    {}

    ### 응답:
    {}"""
    
    conversations = conversations["conversations"]
    texts = []

    for convo in conversations:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(convo[0]["value"], convo[1]["value"]) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

alpaca_dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-korean", split = "train")
alpaca_dataset = alpaca_dataset.train_test_split(test_size=0.1, shuffle=True, seed=seed) # Split dataset into train/validation sets
alpaca_train_set, alpaca_val_set = alpaca_dataset["train"], alpaca_dataset["test"] 

alpaca_train_set = alpaca_train_set.map(formatting_prompts_func, batched = True,)
print(alpaca_train_set[0:2])
alpaca_val_set = alpaca_val_set.map(formatting_prompts_func, batched = True,)
print(alpaca_val_set[0:2])


# Define trainer
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = alpaca_train_set,
    eval_dataset=alpaca_val_set,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,

    args = UnslothTrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,  # training batch size
        per_device_eval_batch_size=BATCH_SIZE,  # validation batch size
        gradient_accumulation_steps = 4, # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps

        # Use warmup_ratio and num_train_epochs for longer runs!
        warmup_ratio = 0.1,
        num_train_epochs = 3,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5, # dummy option now

        # validation and save
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=10,
        save_safetensors=True,
        
        # # callback
        # load_best_model_at_end=True, # this option is only available when eval_strategy == save_strategy
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,

        # dtype
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear", # try "cosine"
        seed = seed,
        output_dir = "outputs/IFT", # Directory for output files
        report_to="tensorboard", # Reporting tool (e.g., TensorBoard, WandB)
    ),
)

trainer_stats = unsloth_train(trainer)

# Save trained model locally and to Hugging Face Hub as normal and quantized form
repo_name = "llama-3.1-8B_lora-IFT_CPT_alpaca-gpt4-kor"
IFT_path = f"{user_name}/{repo_name}"

# Local
model.save_pretrained(repo_name)
tokenizer.save_pretrained(repo_name)

# Online
model.push_to_hub(
    IFT_path, 
    tokenizer=tokenizer,
    private=True,
    token=HF_write_token,
    save_method = "lora", # You can skip this. Default="merged_16bit" prabably. Also available "merged_4bit".
    )
tokenizer.push_to_hub(
    IFT_path, 
    private=True,
    token=HF_write_token,
    )

# GGUF / llama.cpp Conversion
repo_name = "llama-3.1-8B_lora-IFT_CPT_alpaca-gpt4-kor_GGUF"
IFT_GGUF_path = f"{user_name}/{repo_name}"

quantization_method = "q8_0" # or "f16" or "q4_k_m"

model.save_pretrained_gguf(repo_name, tokenizer, quantization_method=quantization_method)
model.push_to_hub_gguf(IFT_GGUF_path, tokenizer, quantization_method=quantization_method, private=True, token=HF_write_token)


'''CPT+IFT Inference Test'''

# Reinit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=IFT_path,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)

# # generative strategy
# temperature=0.7
# top_p = 0.9
# repetition_penalty=1.1
# do_sample=True
# num_beams=1
max_new_tokens=128
use_cache=True

# 1
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(
#     **inputs, 
#     streamer=text_streamer, 
#     temperature=temperature, 
#     top_p=top_p, 
#     do_sample=do_sample,
#     num_beams=num_beams,
#     repetition_penalty=repetition_penalty, 
#     max_new_tokens=max_new_tokens,
#     use_cache=use_cache,
#     )
outputs = model.generate(**inputs, max_new_tokens= max_new_tokens, use_cache=use_cache)
print(tokenizer.batch_decode(outputs)[0])

# 2
alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

    ### 지침:
    {}

    ### 응답:
    {}"""
inputs = tokenizer(
[
    alpaca_prompt.format(
        # "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,", # instruction
        "피보나치 수열을 계속하세요: 1, 1, 2, 3, 5, 8,", # instruction
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens= max_new_tokens, use_cache=use_cache)
print(tokenizer.batch_decode(outputs)[0])

# 3
alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

    ### 지침:
    {}

    ### 응답:
    {}"""
inputs = tokenizer(
[
    alpaca_prompt.format(
        # "Describe the planet Earth extensively.", # instruction
        "지구를 광범위하게 설명하세요.",
        "", # output - leave this blank for generation!
    ),
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens= max_new_tokens, use_cache=use_cache)
print(tokenizer.batch_decode(outputs)[0])

# 4
alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

    ### 지침:
    {}

    ### 응답:
    {}"""
inputs = tokenizer(
[
    alpaca_prompt.format(
        # "What is Korean music like?"
        "한국음악은 어떤가요?", # instruction
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens= max_new_tokens, use_cache=use_cache)
print(tokenizer.batch_decode(outputs)[0])

