import io
import os
import subprocess

import torch
import pretty_midi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import MIDITransformer
from generate import GenerationWorkflow

class TrainVisualizer:
    def __init__(self, log_dir="checkpoints/tensorboard", sample_rate=16000, fps=100,
                 max_audio_seconds=20, ema_decay=0.98, log_interval = 500, generation_args = {}, preload_model: MIDITransformer | None = None):
        """
        ema_decay 用于平滑曲线，例如 avg_loss。
        """

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.sample_rate = sample_rate
        self.fps = fps
        self.max_audio_seconds = max_audio_seconds
        self.ema_decay = ema_decay
        self.log_interval = log_interval
        self._ema_cache = {}  # key -> ema value

        self.preload_model = preload_model

        if "output" not in generation_args:
            generation_args["output"] = "generated.mid"
        self.generation_args = generation_args

    # ---------------------------------------------------------
    # Metric Logging
    # ---------------------------------------------------------
    def log_scalar(self, name, value, step, prefix="train"):
        """
        任何标量都能统一写入 TensorBoard:
            prefix/name
        """
        tag = f"{prefix}/{name}"
        self.writer.add_scalar(tag, value, step)

    def log_lr(self, optimizer, step, prefix="train"):
        """
        支持多 param group 的学习率可视化
        """
        for i, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"{prefix}/lr/group{i}", group["lr"], step)

    def log_loss(self, loss_value, step, prefix="train", name="loss"):
        """
        记录 loss + 平滑后的 EMA loss
        """
        self.writer.add_scalar(f"{prefix}/{name}", loss_value, step)

        key = f"{prefix}/{name}_ema"
        old = self._ema_cache.get(key, loss_value)
        new = self.ema_decay * old + (1 - self.ema_decay) * loss_value
        self._ema_cache[key] = new

        self.writer.add_scalar(f"{prefix}/{name}_ema", new, step)

    def log_grad_norm(self, model, step, prefix="train", name = "grad_norm"):
        """
        自动统计梯度 L2 范数，并记录 mean/max 两个指标。
        """
        total_norm = 0.0
        norms = []

        for p in model.parameters():
            if p.grad is None:
                continue
            norm = p.grad.data.norm(2).item()
            norms.append(norm)

        if len(norms) == 0:
            return

        total_norm = sum(norms)
        mean_norm = np.mean(norms)
        max_norm = np.max(norms)

        self.writer.add_scalar(f"{prefix}/{name}/total", total_norm, step)
        self.writer.add_scalar(f"{prefix}/{name}/mean", mean_norm, step)
        self.writer.add_scalar(f"{prefix}/{name}/max", max_norm, step)

    # ---------------------------------------------------------
    # MIDI Visuals
    # ---------------------------------------------------------
    def generate_and_log_midi(self, step: int, tag="generated"): # 稍稍改进，不过还是临时方案
        GenerationWorkflow(False, self.generation_args, self.preload_model, None)
        # subprocess.run(["sh", "scripts/generate_with_prompt.sh"], check=True)
        self.log_midi(self.generation_args["output"], step, tag)

    def log_midi(self, midi_dir: str, step: int, tag="generated"):
        midi = pretty_midi.PrettyMIDI(midi_dir)
        try:
            self._log_pianoroll(midi, step, tag)
        except Exception as e:
            print(f"[ERROR] Pianoroll logging failed at step {step}: {e}")

        try:
            self._log_audio(midi, step, tag)
        except Exception as e:
            print(f"[ERROR] Audio logging failed at step {step}: {e}")

    def _log_pianoroll(self, midi, step, tag):
        pianoroll = midi.get_piano_roll(fs=self.fps).astype(np.float32)
        if pianoroll.max() > 0:
            pianoroll /= pianoroll.max()

        fig, ax = plt.subplots(figsize=(30, 4), dpi=100)
        ax.imshow(pianoroll, aspect="auto", origin="lower", cmap="gray_r")

        ax.set_ylim(10, 110) # 省略掉一般不会出现的区域
        ax.set_title(f"Pianoroll: {tag}")
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Pitch")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        image_np = plt.imread(buf)
        image = torch.from_numpy(image_np).permute(2, 0, 1)

        with torch.no_grad():
            self.writer.add_image(f"{tag}/pianoroll", image, step)

    def _log_audio(self, midi, step, tag):
        try:
            audio = midi.fluidsynth(fs=self.sample_rate).astype(np.float32)

            max_len = int(self.max_audio_seconds * self.sample_rate)
            if len(audio) > max_len:
                audio = audio[:max_len]

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)

            with torch.no_grad():
                self.writer.add_audio(
                    f"{tag}/audio",
                    audio_tensor,
                    step,
                    sample_rate=self.sample_rate
                )
        except Exception as e:
            print(f"[ERROR] failed logging audio at step {step}:{e}")

    # ---------------------------------------------------------
    def close(self):
        self.writer.close()
