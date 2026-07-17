import re
from collections import defaultdict

class SimpleTokenizer:
    """Simple whitespace-based tokenizer."""

    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4

    def build_vocab(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text):
        tokens = text.lower().split()
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

    def __len__(self):
        return self.vocab_size

class BPETokenizer:
    """Byte Pair Encoding tokenizer (simplified)."""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.merges = []

    def train(self, texts, num_merges=None):
        if num_merges is None:
            num_merges = self.vocab_size - 256
        for text in texts:
            words = text.split()
            for word in words:
                self.word_freqs[' '.join(list(word)) + ' </w>'] += 1
        for _ in range(num_merges):
            pairs = self._get_pairs()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self._merge_vocab(best_pair)
            self.merges.append(best_pair)

    def _get_pairs(self):
        pairs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_vocab(self, pair):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in list(self.word_freqs.keys()):
            new_word = pattern.sub(''.join(pair), word)
            self.word_freqs[new_word] = self.word_freqs.pop(word)

    def encode(self, text):
        return [ord(c) for c in text[:100]]

    def decode(self, tokens):
        return ''.join([chr(t) for t in tokens if t < 128])
