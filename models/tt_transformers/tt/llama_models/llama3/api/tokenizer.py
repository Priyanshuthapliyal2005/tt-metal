class Tokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}

    def encode(self, text: str):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return ''.join(chr(t) for t in tokens)
