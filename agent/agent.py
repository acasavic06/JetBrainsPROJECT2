from transformers import AutoTokenizer,AutoModelForCausalLM
from human_eval.data import read_problems
from human_eval.execution import check_correctness

a=5

model_name= "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")

prompt = "Koliko srbija ima stanovnika?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

print(a)
print("Hello")

