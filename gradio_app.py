import gradio as gr
import torch
from unsloth import FastLanguageModel

model_id = "Ggboykxz/gabon-llm-v1"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def respond(message, history):
    prompt = f"### Instruction:\n{message}\n\n### Réponse:\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response.split("### Réponse:\n")[-1]

gr.ChatInterface(respond, title="GabonLLM v1").launch()
