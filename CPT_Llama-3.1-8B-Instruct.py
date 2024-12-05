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
BATCH_SIZE = 8
rank = 128
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

'''Continued Pre-Training'''

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
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = alpha, # LoRA scaling factor alpha
    lora_dropout = 0.0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = seed,
    use_rslora = True,   # Unsloth supports rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

model.print_trainable_parameters() # the number of trainable weights increase when using "embed_tokens" and "lm_head".

# Dataset formatting function
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    # Wikipedia provides a title and an article text.
    wikipedia_prompt = """위키피디아 기사
    ### 제목: {}

    ### 기사:
    {}"""

    titles = examples["title"]
    texts  = examples["text"]
    outputs = []

    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }

# Load and prepare dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split = "train", )

# Format dataset
train_set = dataset.map(formatting_prompts_func, batched = True,)
print(train_set[0:2])


# Define trainer
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_set,
    dataset_text_field="text",
    max_seq_length=max_seq_length,

    args = UnslothTrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,  # training batch size
        gradient_accumulation_steps = 4, # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps

        # Use warmup_ratio and num_train_epochs for longer runs!
        warmup_ratio = 0.1,
        num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        # validation and save
        logging_steps=100,
        save_strategy='steps',
        save_steps=5000,
        save_total_limit=3,
        save_safetensors=True,

        # dtype
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear", # try "cosine"
        seed = seed,
        output_dir = "outputs/CPT", # Directory for output files
        report_to="tensorboard", # Reporting tool (e.g., TensorBoard, WandB)
    ),
)

trainer_stats = unsloth_train(trainer)

# Save trained model locally and to Hugging Face Hub
repo_name = "CPT_LoRA_Llama-3.1-8B-Instruct-bnb-4bit_wikipedia-ko"
CPT_path = f"{user_name}/{repo_name}"

# Local
model.save_pretrained(repo_name)
tokenizer.save_pretrained(repo_name)

# Online
model.push_to_hub(
    CPT_path, 
    tokenizer=tokenizer,
    private=True,
    token=HF_write_token,
    save_method = "lora", # You can skip this. Default="merged_16bit" prabably. Also available "merged_4bit".
    )
tokenizer.push_to_hub(
    CPT_path, 
    private=True,
    token=HF_write_token,
    )


'''CPT Inference Test'''

# Reinit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CPT_path,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# # generative strategy
# temperature=0.7
# top_p = 0.9
# repetition_penalty=1.1
# do_sample=True
# num_beams=1
max_new_tokens=128
use_cache=True

# Test 1
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