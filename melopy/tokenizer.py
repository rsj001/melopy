import mido
from typing import List, Optional

class MIDITokenizer:
    """
    Granular tokenizer for MIDI events.
    Vocabulary includes:
    - NOTE_ON events (pitch-specific)
    - NOTE_OFF events (pitch-specific)
    - TIME_SHIFT events (quantized time delays)
    - VELOCITY bins (quantized velocities)
    - Special tokens (PAD, BOS, EOS)
    """
    
    def __init__(
        self,
        min_pitch: int = 36,  # C2
        max_pitch: int = 84,  # C7
        num_velocity_bins: int = 32,
        max_time_shift: int = 100,
        time_shift_resolution: int = 10,  # milliseconds per time shift unit
    ):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.num_velocity_bins = num_velocity_bins
        self.max_time_shift = max_time_shift
        self.time_shift_resolution = time_shift_resolution
        
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build the token vocabulary."""
        self.token_to_id = {}
        self.id_to_token = {}
        
        current_id = 0
        
        # Special tokens
        for token in ['<PAD>', '<BOS>', '<EOS>']:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # NOTE_ON tokens (for each pitch in range)
        for pitch in range(self.min_pitch, self.max_pitch + 1):
            token = f'NOTE_ON_{pitch}'
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # NOTE_OFF tokens (for each pitch in range)
        for pitch in range(self.min_pitch, self.max_pitch + 1):
            token = f'NOTE_OFF_{pitch}'
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # TIME_SHIFT tokens
        for shift in range(1, self.max_time_shift + 1):
            token = f'TIME_SHIFT_{shift}'
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # VELOCITY tokens
        for vel_bin in range(self.num_velocity_bins):
            token = f'VELOCITY_{vel_bin}'
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        self.vocab_size = current_id
        self.pad_token_id = self.token_to_id['<PAD>']
        self.bos_token_id = self.token_to_id['<BOS>']
        self.eos_token_id = self.token_to_id['<EOS>']
    
    def velocity_to_bin(self, velocity: int) -> int:
        """Convert MIDI velocity (0-127) to velocity bin."""
        return min(int(velocity / 127 * self.num_velocity_bins), self.num_velocity_bins - 1)
    
    def bin_to_velocity(self, bin_id: int) -> int:
        """Convert velocity bin to MIDI velocity."""
        return int((bin_id + 0.5) * 127 / self.num_velocity_bins)
    
    def encode_midi(self, midi_path: str, piano_channels: Optional[List[int]] = None, pitch_augmentation: int = 0) -> List[int]:
        """
        Encode a MIDI file to a sequence of token IDs.
        
        Args:
            midi_path: Path to MIDI file
            piano_channels: List of MIDI channels to extract (None = all channels)
            
        Returns:
            List of token IDs
        """
        mid = mido.MidiFile(midi_path)
        
        if piano_channels is None:
            piano_channels = list(range(16))
        
        # Track tempo changes for accurate time conversion
        # Start with default MIDI tempo (120 BPM = 500000 microseconds per beat)
        default_tempo = 500000
        tempo_map = [(0, default_tempo)]

        max_pitch = 0
        min_pitch = 100
        
        # First pass: build tempo map from set_tempo events
        for track in mid.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
                if msg.type == 'set_tempo':
                    tempo_map.append((track_time, msg.tempo))
                    # print(f"Tick #{track_time} Tempo: {msg.tempo} ms/beat")
                if msg.type == 'note_on' and msg.velocity > 0:
                    max_pitch = max(max_pitch, msg.note)
                    min_pitch = min(min_pitch, msg.note)

        def legal_interval(min_p, max_p):
            return max_pitch <= self.max_pitch and min_pitch >= self.min_pitch
        
        pitch_offset = 0
        if not legal_interval(min_pitch, max_pitch):
            if legal_interval(min_pitch - 8, max_pitch - 8):
                pitch_offset = -8
            elif legal_interval(min_pitch + 8, max_pitch + 8):
                pitch_offset = 8
            else:
                pitch_offset =  ((self.max_pitch + self.min_pitch) - (max_pitch + min_pitch)) // 2

        if pitch_augmentation != 0:
            if abs(pitch_augmentation) <= 3 or legal_interval(min_pitch + pitch_offset + pitch_augmentation, max_pitch + pitch_offset + pitch_augmentation):
                pitch_offset += pitch_augmentation
            else:
                return []
        
        # Sort by time in case tracks have tempo events at different positions
        tempo_map.sort(key=lambda x: x[0])
        
        def ticks_to_ms(ticks: int, ticks_per_beat: int) -> int:
            """Convert MIDI ticks to milliseconds using tempo map."""
            if ticks == 0:
                return 0
            
            ms = 0
            current_ticks = 0
            current_tempo = tempo_map[0][1]
            
            for i, (tempo_ticks, new_tempo) in enumerate(tempo_map):
                if tempo_ticks >= ticks:
                    delta_ticks = ticks - current_ticks
                    ms += int(delta_ticks * current_tempo / ticks_per_beat / 1000)
                    return ms
                else:
                    delta_ticks = tempo_ticks - current_ticks
                    ms += int(delta_ticks * current_tempo / ticks_per_beat / 1000)
                    current_ticks = tempo_ticks
                    current_tempo = new_tempo
            
            delta_ticks = ticks - current_ticks
            ms += int(delta_ticks * current_tempo / ticks_per_beat / 1000)
            return ms
        
        # Extract note events with absolute timing
        events = []
        
        for track in mid.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time

                if msg.type == 'note_on' and msg.channel in piano_channels:
                    new_pitch = msg.note + pitch_offset
                    if self.min_pitch <= new_pitch <= self.max_pitch:
                        events.append({
                            'type': 'note_on' if msg.velocity > 0 else 'note_off',
                            'pitch': new_pitch,
                            'velocity': msg.velocity,
                            'time_ticks': track_time
                        })
                elif msg.type == 'note_off' and msg.channel in piano_channels:
                    new_pitch = msg.note + pitch_offset
                    if self.min_pitch <= new_pitch <= self.max_pitch:
                        events.append({
                            'type': 'note_off',
                            'pitch': new_pitch,
                            'velocity': 0,
                            'time_ticks': track_time
                        })
        
        # Sort events by time_ticks
        events.sort(key=lambda x: x['time_ticks'])
        
        tokens = [self.bos_token_id]
        prev_time_ms = 0
        
        for event in events:
            # Convert ticks to milliseconds
            event_time_ms = ticks_to_ms(event['time_ticks'], mid.ticks_per_beat)
            time_diff_ms = event_time_ms - prev_time_ms
            
            # Emit time shift tokens if needed
            if time_diff_ms > 0:
                time_units = time_diff_ms // self.time_shift_resolution
                
                # shorten too long time_units
                if time_units > self.max_time_shift * 5:
                    time_units = self.max_time_shift * 5
                
                while time_units > 0:
                    shift = min(time_units, self.max_time_shift)
                    tokens.append(self.token_to_id[f'TIME_SHIFT_{shift}'])
                    time_units -= shift
            
            # Emit velocity token for note_on events
            if event['type'] == 'note_on' and event['velocity'] > 0:
                vel_bin = self.velocity_to_bin(event['velocity'])
                tokens.append(self.token_to_id[f'VELOCITY_{vel_bin}'])
                tokens.append(self.token_to_id[f'NOTE_ON_{event["pitch"]}'])
            elif event['type'] == 'note_off':
                tokens.append(self.token_to_id[f'NOTE_OFF_{event["pitch"]}'])
            
            prev_time_ms = event_time_ms
        
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode_to_midi(self, token_ids: List[int], output_path: str, tempo: int = 500000):
        """
        Decode token IDs to a MIDI file.
        
        Args:
            token_ids: List of token IDs
            output_path: Path to save MIDI file
            tempo: Microseconds per quarter note
        """
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        
        current_time = 0
        current_velocity = 64  # Default velocity
        
        for token_id in token_ids:
            if token_id >= self.vocab_size:
                continue
                
            token = self.id_to_token[token_id]
            
            if token.startswith('TIME_SHIFT_'):
                shift = int(token.split('_')[-1])
                current_time += shift * self.time_shift_resolution
            
            elif token.startswith('VELOCITY_'):
                vel_bin = int(token.split('_')[-1])
                current_velocity = self.bin_to_velocity(vel_bin)
            
            elif token.startswith('NOTE_ON_'):
                pitch = int(token.split('_')[-1])
                beats = (current_time / 1000) / (tempo / 1_000_000)  # current_time -> beats formula
                ticks = int(beats * mid.ticks_per_beat)
                track.append(mido.Message('note_on', note=pitch, velocity=current_velocity, time=ticks, channel=0))
                current_time = 0

            elif token.startswith('NOTE_OFF_'):
                pitch = int(token.split('_')[-1])
                beats = (current_time / 1000) / (tempo / 1_000_000)
                ticks = int(beats * mid.ticks_per_beat)
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=ticks, channel=0))
                current_time = 0
        
        mid.save(output_path)
    
    def __len__(self):
        return self.vocab_size
