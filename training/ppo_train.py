# ppo_train.py (pseudocode)
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
policy = PeftModel.from_pretrained(base, "sft_lora_out")  # LoRA adapter loaded

ppo_config = PPOConfig(model_name=MODEL, batch_size=16, ppo_epochs=4, ...)
ppo_trainer = PPOTrainer(ppo_config, model=policy, tokenizer=tokenizer)

def env_step(url):
    # 1) call navigator API to get markdown+metadata
    page = http_get(f"http://navigator/navigate?url={url}")
    prompt = build_prompt(page["markdown"], page["metadata"])
    return prompt

for update in range(N_updates):
    prompts = [env_step(url) for url in sample_urls(batch_size)]
    responses = ppo_trainer.generate(prompts, max_length=512)
    rewards = []
    for url, resp in zip(sample_urls, responses):
        # validate JSON
        valid, parsed = try_parse_json(resp)
        if not valid:
            r = 0
        else:
            # check via checker service (could simulate or actually interact)
            check = http_post("http://checker/check", json={"url": url, "parsed_json": parsed})
            r = 1 if check["success"] else 0
        rewards.append(r)
    # PPO step: updates LoRA params using rewards
    ppo_trainer.step(prompts, responses, rewards)
