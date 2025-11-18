import mido
from typing import List, Optional
from mido import MidiFile, MidiTrack, MetaMessage, bpm2tempo

const_log_bin = [0, 5, 10, 15] + \
    [i for i in range(20, 400, 10)] + \
        [i for i in range(400, 1000, 40)] + \
            [i for i in range(1000, 2000, 100)] + \
                [i for i in range(2000, 3000, 200)] + \
                    [i for i in range(3000, 4200, 300)] + \
                        [i for i in range(4200, 5000, 400)] + \
                        [5000, 5500, 6000]

const_log_bin_2 = [i for i in range(100, 1500, 50)] + \
                [i for i in range(1500, 2000, 100)] + \
                [i for i in range(2000, 3000, 250)] + \
                [3000, 4000, 5000, 6000]

class MIDITokenizer:
    """
    Granular tokenizer for MIDI events.
    Vocabulary includes:
    - NOTE_PITCH events (pitch-specific)
    - TIME_SHIFT events (quantized time delays)
    - VELOCITY bins (quantized velocities)
    - DURATION bins (quantized note durations)
    - Special tokens (PAD, BOS, EOS)
    """
    
    def __init__(
        self,
        min_pitch: int = 12,  # C0
        max_pitch: int = 109,  # C9
        velocity_bins: List[int] = [i for i in range(1, 128, 4)],
        duration_bins: List[int] = const_log_bin_2,
        time_shift_bins: List[int] = const_log_bin,
    ):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.velocity_bins = velocity_bins
        self.duration_bins = duration_bins
        self.time_shift_bins = time_shift_bins
        self.version = "1.3"

        self.token_to_id = {}
        self.id_to_token = {}
        current_id = 0
        # Special tokens

        self.special_token = ['<PAD>', '<BOS>', '<EOS>', '<MASK>']

        self.vocab_full = {"note": self.special_token + [f'NOTE_{pitch}' for pitch in range(self.min_pitch, self.max_pitch + 1)],
                      "duration": self.special_token + [f'DURATION_{dur_bin}' for dur_bin in duration_bins],
                      "velocity": self.special_token + [f'VELOCITY_{vel_bin}' for vel_bin in velocity_bins],
                      "time_shift": self.special_token + [f'TIME_SHIFT_{shift}' for shift in time_shift_bins],
                      }
                      
        for category, tokens in self.vocab_full.items():
            self.token_to_id[category] = {}
            self.id_to_token[category] = {}
            current_id = 0
            for token in tokens:
                self.token_to_id[category][token] = current_id
                self.id_to_token[category][current_id] = token
                current_id += 1
        
        self.vocab_size = {cat: len(toks) for cat, toks in self.vocab_full.items()}
        self.vocab_size_full = sum(self.vocab_size.values())

        self.bos_token = (self.token_to_id['note']['<BOS>'],
                    self.token_to_id['duration']['<BOS>'],
                    self.token_to_id['velocity']['<BOS>'],
                    self.token_to_id['time_shift']['<BOS>'])
        self.eos_token = (self.token_to_id['note']['<EOS>'],
                    self.token_to_id['duration']['<EOS>'],
                    self.token_to_id['velocity']['<EOS>'],
                    self.token_to_id['time_shift']['<EOS>'])
        self.pad_token = (self.token_to_id['note']['<PAD>'],
                    self.token_to_id['duration']['<PAD>'],
                    self.token_to_id['velocity']['<PAD>'],
                    self.token_to_id['time_shift']['<PAD>'])

    def unify_bpm_by_ticks(self, in_path: str, target_bpm: float = 120) -> MidiFile:
        mid = MidiFile(in_path)
        tpb = mid.ticks_per_beat
        target_tempo = bpm2tempo(target_bpm)
        # ----------------------------------------------------------
        # 1. Build tempo map (absolute_ticks, tempo)
        # ----------------------------------------------------------
        tempo_events = [(0, 500000)]   # default tempo
        for track in mid.tracks:
            abs_t = 0
            for msg in track:
                abs_t += msg.time
                if msg.type == "set_tempo":
                    tempo_events.append((abs_t, msg.tempo))
        tempo_events.sort()
        # ----------------------------------------------------------
        # Build cumulative seconds at each tempo change
        # ----------------------------------------------------------
        segs = []
        cum_seconds = 0.0
        for i in range(len(tempo_events)):
            t0, tempo = tempo_events[i]
            if i + 1 < len(tempo_events):
                t1 = tempo_events[i + 1][0]
            else:
                t1 = None  # until infinity
            segs.append((t0, t1, tempo, cum_seconds))
            if t1 is not None:
                dt_ticks = t1 - t0
                cum_seconds += dt_ticks * tempo / (1e6 * tpb)
        # ----------------------------------------------------------
        def ticks_to_seconds(tick):
            """Fast piecewise conversion"""
            for t0, t1, tempo, base_sec in segs:
                if t1 is None or tick < t1:
                    return base_sec + (tick - t0) * tempo / (1e6 * tpb)
            raise ValueError("out of range, segs empty?")
        # ----------------------------------------------------------
        def seconds_to_ticks(sec):
            """Constant-tempo conversion"""
            return int(round(sec * 1e6 * tpb / target_tempo))
        # ----------------------------------------------------------
        # 2. For each track convert absolute ticks → seconds → new ticks
        # ----------------------------------------------------------
        new_tracks = []
        for track in mid.tracks:
            abs_t = 0
            events = []
            for msg in track:
                abs_t += msg.time
                sec = ticks_to_seconds(abs_t)
                new_abs = seconds_to_ticks(sec)
                events.append((new_abs, msg))
            # rebuild track with fresh delta times
            events.sort(key=lambda x: x[0])
            nt = MidiTrack()
            last = 0
            for abs_tk, msg in events:
                dt = abs_tk - last
                last = abs_tk
                # remove tempo events; we will add our own
                if msg.type == "set_tempo":
                    continue
                nt.append(msg.copy(time=dt))
            new_tracks.append(nt)
        # ----------------------------------------------------------
        # 3. Build output MIDI
        # ----------------------------------------------------------
        out = MidiFile(ticks_per_beat=tpb)
        for i, tr in enumerate(new_tracks):
            nt = MidiTrack()
            if i == 0:
                nt.append(MetaMessage("set_tempo", tempo=target_tempo, time=0))
            nt.extend(tr)
            out.tracks.append(nt)

        return out
    def encode_midi(self, midi_path: str, piano_channels: Optional[List[int]] = None) -> List[tuple]:
        """
        Encode a MIDI file to a sequence of token IDs.
        
        Args:
            midi_path: Path to MIDI file
            piano_channels: List of MIDI channels to extract (None = all channels)
            
        Returns:
            List of token IDs
        """

        # assume 120 BPM preprocessing is done elsewhere
        mid = self.unify_bpm_by_ticks(midi_path, 120)

        zoom_ratio = 480.0 / mid.ticks_per_beat
        if mid.ticks_per_beat == 480:
            zoom_ratio = 1
        
        if piano_channels is None:
            piano_channels = list(range(16))
        
        events = []
        for track in mid.tracks:
            note_last_on = [-1] * 130
            abs_time = 0
            for msg in track:
                abs_time += msg.time * zoom_ratio
                if msg.type not in ["note_on", "note_off"]:
                    pass
                elif msg.velocity == 0 or msg.type == 'note_off':
                    if msg.channel not in piano_channels:
                        continue
                    pitch = msg.note
                    if note_last_on[pitch] != -1:
                        events[note_last_on[pitch]][1] = abs_time - events[note_last_on[pitch]][1]
                        note_last_on[pitch] = -1
                else:
                    if msg.channel not in piano_channels:
                        continue
                    pitch = msg.note
                    events.append([pitch, abs_time, msg.velocity, abs_time])
                    if note_last_on[pitch] != -1:
                        events[note_last_on[pitch]][1] = abs_time - events[note_last_on[pitch]][1]
                    note_last_on[pitch] = len(events) - 1

        events.sort(key=lambda x: x[3])
        for i in range(len(events)-1,0,-1):
            events[i][3] = events[i][3] - events[i - 1][3]
        
        maxm = 0
        maxd = 0
        def quantize(value, bins):
            return bins[min(range(len(bins)), key=lambda x: abs(bins[x]-value))]
        for i in range(len(events)):
            maxm = max(maxm, events[i][3])
            maxd = max(maxd, events[i][1])
            events[i][1] = quantize(events[i][1], self.duration_bins)
            events[i][2] = quantize(events[i][2], self.velocity_bins)
            events[i][3] = quantize(events[i][3], self.time_shift_bins)
        # print("max delta time:", maxm)
        # print("max duration:", maxd)

        token_ids = [self.bos_token]
        for event in events:
            pitch, duration, velocity, delta_time = event
            if pitch < self.min_pitch or pitch > self.max_pitch:
                continue
            token_ids.append((
                self.token_to_id['note'][f'NOTE_{pitch}'],
                self.token_to_id['duration'][f'DURATION_{duration}'],
                self.token_to_id['velocity'][f'VELOCITY_{velocity}'],
                self.token_to_id['time_shift'][f'TIME_SHIFT_{delta_time}']
            ))
        token_ids.append(self.eos_token)
        return token_ids
    
    def decode_to_midi(self, token_ids: List[tuple], output_path: str):
        """
        Decode a sequence of token IDs back to a MIDI file.
        
        Args:
            token_ids: List of token IDs
            output_path: Path to save the decoded MIDI file
        """
        mid = mido.MidiFile()
        track = mido.MidiTrack()

        events = []
        for token in token_ids:
            pitch, duration, velocity, delta_time = token
            if self.id_to_token['note'][pitch] == '<EOS>':
                break
            if self.id_to_token['note'][pitch][0] == '<': # HARDCODE
                continue

            pitch = int(self.id_to_token['note'][pitch].split('_')[-1])
            duration = int(self.id_to_token['duration'][duration].split('_')[-1])
            velocity = int(self.id_to_token['velocity'][velocity].split('_')[-1])
            delta_time = int(self.id_to_token['time_shift'][delta_time].split('_')[-1])
            events.append((pitch, duration, velocity, delta_time))

        rev = []
        abs_time = 0
        for i in events:
            pitch, duration, velocity, delta_time = i
            abs_time += delta_time
            rev.append((pitch, duration, velocity, abs_time))
            rev.append((pitch, duration, 0, abs_time + duration))
        rev.sort(key=lambda x: x[3])

        prev_time = 0
        for token in rev:
            pitch, duration, velocity, abs_time = token
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=abs_time-prev_time))
            prev_time = abs_time
        mid.tracks.append(track)
        mid.save(output_path)
    def __len__(self):
        return self.vocab_size