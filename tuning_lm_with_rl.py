import os

import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()


@dataclass
# 定义了训练的参数ScriptArguments,包括模型名称、数据集、学习率等超参数
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="./checkpoints/tuning_llama_rl/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
# build_dataset构建了训练数据集
def build_dataset(
        tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    train_dataset = load_dataset(dataset_name, split="train")
    original_columns = train_dataset.column_names
    num_proc = 24

    # 定义了preprocess_function来预处理每个sample
    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        # 对于每个sample中的question,构建query字符串:"Question: {question}\n\nAnswer:"
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)  # 用tokenizer对query字符串进行tokenize,获取input_ids
            new_examples["query"].append(query) # 将处理过的query和input_ids保存到新的样本中
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples
    
    # 使用map对数据集批量应用preprocess_function进行预处理
    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )  
    # 使用filter过滤掉序列长度超过max_length的样本
    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.max_length, batched=False)

    ds.set_format(type="torch")     # 设置dataset的格式为PyTorch的tensor格式
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


reward_model_name = script_args.reward_model_name
# 定义PPO的配置PPOConfig
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
rw_kwargs = {
    "return_all_scores": True,  # 返回每个token的情感分数,而不只是句子总体分数
    "function_to_apply": "none",    # 不应用任何预处理函数
    "batch_size": 16,   # 调用pipeline时的批处理大小为16
    "truncation": True  # 是否截断文本以适应模型最大长度
}

# 根据模型不同,加载对应的tokenizer,如果是decapoda的Llama,需要添加特殊token
if "decapoda" in script_args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
# 对于其他模型,如果没有定义pad token,则将eos token赋值给pad token
else:   
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, script_args.dataset_name)

# 获取当前进程的device id,用于后续模型并行.
current_device = Accelerator().local_process_index
# 定义LoRA的配置,包括压缩率r、α参数等
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 用AutoModelForCausalLMWithValueHead加载语言模型,这是在预训练LM基础上添加了价值头的模型
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,  # 设置load_in_8bit=True,可以加速推理速度
    device_map={"": current_device},    # device_map用于在分布式环境下设置模型放置的设备
    peft_config=lora_config,
)

# 构建优化器,这里根据参数可以选择Adam或Adafactor
optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
# 用PPOTrainer封装PPO的训练,包括数据加载、生成样本、计算rewards等
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
# 构建奖励模型pipeline,这里使用文本分类模型,可以替换成其他的奖励函数
reward_model = pipeline(
    "text-classification",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# 定义生成文本的超参数generation_kwargs,如生成长度范围,会传到模型的generate函数
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# 在循环中,生成response,计算奖励,执行PPO的训练步骤,更新模型
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    # 用PPOTrainer生成回复文本response
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # 计算文本的情感得分作为奖励reward
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]  # 构建文本为prompt+response的组合,传给奖励模型reward_model计算奖励
    reward_outputs = reward_model(texts, **rw_kwargs)   # 从奖励模型输出中取出分数,做一定调整获得reward
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in reward_outputs]

    # 调用PPOTrainer的step方法,传入文本、response和reward,执行PPO的训练步骤,更新模型
    '''
    target_kl这个early stopping的参数,其在PPO训练代码中的执行逻辑如下:

        在PPO算法的训练流程中,每次更新策略网络后,会计算新的策略与旧策略的KL散度。
        然后将计算的KL散度与预先定义的target_kl进行比较。
        如果KL散度大于target_kl,则说明策略网络更新太大,需要early stop当前epoch。
        具体来说,是通过抛出一个用于early stop的异常来结束训练循环。
        如果KL散度小于target_kl,则继续正常的PPO训练流程。
        这样就使用target_kl来控制策略网络更新的最大程度。
        所以 target_kl 作为一个阈值,通过比较KL散度的大小来确定是否early stop当前epoch。

    这通常是在PPO训练器的step函数中实现,计算KL后与target_kl比较,来决定是否early stop。

    '''
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)    # 记录训练过程的统计数据,如loss等

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
