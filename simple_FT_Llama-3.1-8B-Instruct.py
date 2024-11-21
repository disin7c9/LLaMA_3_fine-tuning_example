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
BATCH_SIZE = 2
rank = 16
alpha = rank*2

# designate base model
base_model_path = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
others = [
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-1B-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Meta-Llama-3.1-8B",
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
]

'''Fine-tuning'''

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=load_in_4bit,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = rank, # LoRA rank. Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], 
    lora_alpha = alpha, # LoRA scaling factor alpha
    lora_dropout = 0.0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = seed,
    use_rslora = False,   # Unsloth supports rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

model.print_trainable_parameters()

# Dataset formatting function, You can also use unsloth.get_chat_template function to get the correct chat template.
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
alpaca_dataset = alpaca_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42) # Split dataset into train/validation sets
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
        gradient_accumulation_steps = 16, # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps

        # Use warmup_ratio and num_train_epochs for longer runs!
        warmup_ratio = 0.1,
        num_train_epochs = 2,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 1e-4,

        # validation and save
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=1000,
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
        weight_decay = 0.01,
        lr_scheduler_type = "linear", # try "cosine"
        seed = seed,
        output_dir = "outputs/lora", # Directory for output files
        report_to="tensorboard", # Reporting tool (e.g., TensorBoard, WandB)
    ),
)

trainer_stats = unsloth_train(trainer)

# Save trained model locally and to Hugging Face Hub as normal and quantized form
repo_name = "LoRA_Llama-3.1-8B-Instruct-bnb-4bit_alpaca-gpt4-ko"
repo_path = f"{user_name}/{repo_name}"

# Local
model.save_pretrained(repo_name)
tokenizer.save_pretrained(repo_name)

# Online
model.push_to_hub(
    repo_path, 
    tokenizer=tokenizer,
    private=True,
    token=HF_write_token,
    save_method = "lora", # You can skip this. Default="merged_16bit" prabably. Also available "merged_4bit".
    )
tokenizer.push_to_hub(
    repo_path, 
    private=True,
    token=HF_write_token,
    )


# GGUF / llama.cpp Conversion
repo_name = "LoRA_Llama-3.1-8B-Instruct-bnb-4bit_alpaca-gpt4-ko-GGUF"
repo_GGUF_path = f"{user_name}/{repo_name}"

quantization_method = "q8_0" # or "f16" or "q4_k_m"

model.save_pretrained_gguf(repo_name, tokenizer, quantization_method=quantization_method)
model.push_to_hub_gguf(repo_GGUF_path, tokenizer, quantization_method=quantization_method, private=True, token=HF_write_token)


'''Inference Test'''

# Reinit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = rank, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = alpha,
)

model = model.bfloat16() # fit the loaded lora dtype
model.load_adapter(repo_path)
model.merge_and_unload()

FastLanguageModel.for_inference(model) # Enable native 2x faster inference


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

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# 2
alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청을 완벽하게 완료하는 응답을 작성하세요.

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

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs)[0])

# 3
alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청에 적절하게 응답하세요.

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

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
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

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

