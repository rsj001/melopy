from melopy.tokenizer import MIDITokenizer
from rich import print

def test_new_tokenizer():
    tokenizer = MIDITokenizer()
    assert tokenizer.vocab_size_full == sum(tokenizer.vocab_size.values())
    print("\n[blue] Vocabulary sizes by category:", tokenizer.vocab_size)

def test_token():
    tokenizer = MIDITokenizer()
    tokens = tokenizer.encode_midi("177229 - Summer.mid")
    print("\n[blue] Tokenization successful.")
    print(tokens)
    tokenizer.decode_to_midi(tokens, "reconstructed.mid")
    print("\n[blue] De-Tokenization successful.")