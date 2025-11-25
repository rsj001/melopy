import sys
sys.path.append("melopy")
from generate import GenerationWorkflow

import gradio as gr
import pretty_midi

# 加载模型
model = None

# 包装成 gradio 调用
def infer(prompt_midi,
    prompt_length,
    max_length,
    temperature,
    top_k,
    top_p,
    seed,
    piano_channels):
    sample_rate = 44100
    args = {
        "checkpoint_dir": "checkpoints_ex", # Change this when using
        "checkpoint_name": "checkpoint_quicksave.pt",
        "output": "results/app/generated.mid",
        "prompt_midi": prompt_midi.name if prompt_midi else None, 
        "prompt_length": prompt_length, 
        "max_length": max_length, 
        "temperature": temperature, 
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed, 
        "piano_channels": piano_channels
    }
    prompt = None
    model = None
    GenerationWorkflow(False, args, model, prompt)
    midi = pretty_midi.PrettyMIDI(args["output"])
    audio = midi.fluidsynth(fs=sample_rate).astype('float32')
    return (sample_rate, audio)

default = {
    "prompt_midi": None, 
    "prompt_length": None, 
    "max_length": 512, 
    "temperature": 1.1, 
    "top_k": 50,
    "top_p": 0.95,
    "seed": None, 
    "piano_channels": '0, 1, 2, 3, 4, 5' 
}

inputs = [
    gr.File(label="Prompt MIDI (Optional)", file_types=[".mid"]),
    gr.Number(label="Prompt Length", value=default["prompt_length"]),
    gr.Number(label="Max Length", value=default["max_length"]),
    gr.Slider(0.1, 2.0, value=default["temperature"], step=0.1, label="Temperature"),
    # gr.Textbox(label="topk"),
    # gr.Textbox(label="topp"),
    gr.Number(label="Top K", value=(default["top_k"])),
    gr.Slider(0.01, 1.00, value=default["top_p"], step=0.01, label="Top P"),
    gr.Number(label="Seed", value=default["seed"]),
    gr.Textbox(label="Piano Channels", value=default["piano_channels"])
]

outputs = [
    gr.Audio(label="Audio",type="numpy")
]

demo = gr.Interface(fn=infer, inputs=inputs, outputs=outputs, title="Melopy Demo")

if __name__ == "__main__":
    demo.launch()