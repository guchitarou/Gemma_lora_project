import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from utils import create_conversation

# BitsAndBytesConfig → 4bit量子化の設定
# prepare_model_for_kbit_training → LoRA 学習用に量子化モデルを調整

model_id ="./model_weights"
output_dir_path = "./gemma-goku-prj"
dataset = load_dataset("csv", data_files="training_data_goku_1k.csv", encoding="cp932")

dataset = dataset.map(create_conversation, remove_columns=["No","Question", "Answer"])
# DatasetDict -> Dataset に変換(train split を取り出す)
dataset = dataset["train"]
# split dataset into 90% training samples and 10% test samples
dataset = dataset.train_test_split(test_size=0.1, shuffle=False)

if torch.cuda.is_bf16_supported():
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

model_kwargs = dict(
    dtype=torch_dtype,
    device_map="auto",
)

# 4bit量子化して VRAM を大幅に節約する。
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=torch_dtype,
)

model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")

if (torch.cuda.get_device_properties(0).total_memory/1024**3) > 16:
    model = prepare_model_for_kbit_training(model)


# loraの設定
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    # no target_modules — PEFT's Gemma 4 defaults scope to the LM layers
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"], # make sure to save the lm_head and embed_tokens as you train the special tokens
    ensure_weight_tying=True,
)


# LoRA で学習するための「SFT（Supervised Fine-Tuning）全体設定
args = SFTConfig(
    output_dir=output_dir_path,         # directory to save and repository id
    max_length=512,                         # max length for model and packing of the dataset
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    per_device_eval_batch_size=1,           # batch size per device during evaluation
    optim="paged_adamw_8bit",              # adamw_torch_fusedを使っていたが、一旦paged_adamw_8bitを使用
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=2e-4,                     # learning rate
    fp16=True if torch_dtype == torch.float16 else False,  # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False, # use bfloat16 precision
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={"skip_prepare_dataset": True}, # important for collator
    remove_unused_columns = False,                 # important for collator
)

# LoRA 学習を成立させるためのデータ整形関数
def collate_fn(examples):
    texts = []

    for example in examples:
        full_text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(full_text.strip())

    # Tokenize the texts and process the audios
    batch = processor(text=texts, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and audio tokens in the loss computation
    labels = batch["input_ids"].clone()

    target_tokens = [
        processor.tokenizer.convert_tokens_to_ids("<|turn>"),
        processor.tokenizer.convert_tokens_to_ids("model"),
        processor.tokenizer.convert_tokens_to_ids("\n")
    ]
    target_len = len(target_tokens)

    for i in range(labels.size(0)):
        row_tokens = batch["input_ids"][i].tolist()

        # Find where the assistant block begins
        assistant_start_idx = None
        for idx in range(len(row_tokens) - target_len + 1):
            if row_tokens[idx : idx + target_len] == target_tokens:
                # We want to keep loss calculation on the assistant transcription tokens,
                # so we move the index right past the assistant header ('<|turn>\nmodel\n')
                assistant_start_idx = idx + target_len
                break

        if assistant_start_idx is not None:
            # Mask everything from index 0 up to the start of the actual Japanese text response
            labels[i, :assistant_start_idx] = -100
        else:
            # Fallback safety: if template matching fails for an anomalous row, mask padding anyway
            print("WARNING: maybe the sample is too long, try to increase `token_limit` value.")
            labels[i, labels[i] == processor.tokenizer.pad_token_id] = -100


    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    return batch


# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()


# free the memory again
del model
del trainer
torch.cuda.empty_cache()
