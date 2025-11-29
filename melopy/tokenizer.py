from miditok import Octuple, TokenizerConfig  # here we choose to use REMI

# Our parameters
TOKENIZER_PARAMS = {
    "pitch_range": (12, 109),
    "beat_res": {(0, 4): 32},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": True,
    "num_tempos": 96,  # number of tempo bins
    "tempo_range": (20, 400),  # (min, max) 20 + 95 * 4 = 400
    "max_bar_embedding": 250,
    # "sustain_pedal_duration": True,
    # "remove_duplicated_notes": True,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
class MIDITokenizer:
    def __init__(self):
        # Creates the tokenizer
        self.tokenizer = Octuple(config)
        self.tokenizer._verbose = False

        self.categories = ["Pitch","Pos","Bar","Velocity","Duration","Program","Tempo"]
        self.num_tasks = len(self.categories)
        self.vocab_sizes = self.tokenizer.len

        self.bos_token = [1] * self.num_tasks
        self.eos_token = [2] * self.num_tasks
        self.pad_token = [0] * self.num_tasks
    def encode_midi(self, midi_path: str):
        return self.tokenizer(midi_path)
    
    def decode_to_midi(self, token_ids, output_path: str):
        midi = self.tokenizer(token_ids)
        midi.dump_midi(output_path)