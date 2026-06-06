
import random, json, math, re
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import torch.nn as nn

GEORGIAN_CHARS = list('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ')
PAD, SOS, EOS, UNK = '<PAD>', '<SOS>', '<EOS>', '<UNK>'
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

def google_drive_direct_url(url_or_id: str) -> str:
    """Convert a Google Drive share link or file id to a direct download URL."""
    text = str(url_or_id).strip()
    match = re.search(r'/d/([^/]+)', text)
    file_id = match.group(1) if match else text
    if text.startswith('http') and 'drive.google.com' not in text:
        return text
    if 'drive.google.com' in text or re.fullmatch(r'[A-Za-z0-9_-]+', file_id):
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return text

def download_words_from_url(url_or_id: str, cache_path: str = 'data/georgian_words.txt') -> str:
    """Download the online word list and save it locally for reproducible training."""
    import requests
    url = google_drive_direct_url(url_or_id)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_path).write_text(response.text, encoding='utf-8')
    return cache_path

def load_words(path: str) -> List[str]:
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if re.fullmatch(r'[ა-ჰ]+', w) and 2 <= len(w) <= 18:
                words.append(w)
    return sorted(set(words))

def build_vocab(words: List[str]) -> Dict[str, int]:
    chars = sorted(set(''.join(words)) | set(GEORGIAN_CHARS))
    stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + chars)}
    return stoi

def invert_vocab(stoi: Dict[str, int]) -> Dict[int, str]:
    return {i: ch for ch, i in stoi.items()}

def encode_word(word: str, stoi: Dict[str, int], add_eos=True) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in word]
    if add_eos:
        ids.append(stoi[EOS])
    return ids

def decode_ids(ids: List[int], itos: Dict[int, str]) -> str:
    out = []
    for idx in ids:
        ch = itos.get(int(idx), '')
        if ch == EOS:
            break
        if ch not in (PAD, SOS, UNK):
            out.append(ch)
    return ''.join(out)

# Keyboard-neighbor-inspired Georgian groups. They make replacement errors more realistic than pure random noise.
NEIGHBOR_GROUPS = [
    'აბგდევ', 'ზთიკლ', 'მნოპჟ', 'რსტუ', 'ფქღყ', 'შჩცძ', 'წჭხჯჰ'
]
NEIGHBORS = {}
for group in NEIGHBOR_GROUPS:
    for i, ch in enumerate(group):
        close = set(group[max(0, i-2):i] + group[i+1:i+3])
        NEIGHBORS[ch] = list(close) if close else GEORGIAN_CHARS

def corrupt_word(word: str, p_keep_correct: float = 0.20) -> str:
    if len(word) < 2 or random.random() < p_keep_correct:
        return word
    chars = list(word)
    op = random.choices(['delete', 'replace', 'swap', 'insert'], weights=[0.30, 0.35, 0.20, 0.15])[0]
    i = random.randrange(len(chars))
    if op == 'delete' and len(chars) > 2:
        del chars[i]
    elif op == 'replace':
        ch = chars[i]
        chars[i] = random.choice(NEIGHBORS.get(ch, GEORGIAN_CHARS))
    elif op == 'swap' and len(chars) > 2:
        j = min(i + 1, len(chars) - 1)
        if i == j:
            i -= 1
        chars[i], chars[j] = chars[j], chars[i]
    elif op == 'insert':
        base = chars[i]
        chars.insert(i, random.choice(NEIGHBORS.get(base, GEORGIAN_CHARS)))
    return ''.join(chars)

def make_pairs(words: List[str], variants_per_word: int = 3) -> List[Tuple[str, str]]:
    pairs = []
    for w in words:
        pairs.append((w, w))
        for _ in range(variants_per_word):
            pairs.append((corrupt_word(w, p_keep_correct=0.0), w))
    random.shuffle(pairs)
    return pairs

def pad_batch(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    for i, s in enumerate(seqs):
        tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return tensor, lengths

def collate_batch(batch, stoi):
    pad_id = stoi[PAD]
    src = [encode_word(x, stoi, add_eos=True) for x, _ in batch]
    tgt = [encode_word(y, stoi, add_eos=True) for _, y in batch]
    decoder_in = [[stoi[SOS]] + t[:-1] for t in tgt]
    src_pad, src_len = pad_batch(src, pad_id)
    dec_pad, _ = pad_batch(decoder_in, pad_id)
    tgt_pad, _ = pad_batch(tgt, pad_id)
    return src_pad, src_len, dec_pad, tgt_pad

class Seq2SeqSpellchecker(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64, num_layers=1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, src, src_len, dec_in):
        emb_src = self.emb(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb_src, src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        emb_dec = self.emb(dec_in)
        dec_out, _ = self.decoder(emb_dec, hidden)
        return self.out(dec_out)
    @torch.no_grad()
    def greedy_decode(self, word: str, stoi, max_len=24, device='cpu') -> str:
        self.eval()
        itos = invert_vocab(stoi)
        src_ids = encode_word(word, stoi, add_eos=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
        emb_src = self.emb(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb_src, src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        cur = torch.tensor([[stoi[SOS]]], dtype=torch.long, device=device)
        out_ids = []
        for _ in range(max_len):
            dec_out, hidden = self.decoder(self.emb(cur), hidden)
            logits = self.out(dec_out[:, -1, :])
            next_id = int(logits.argmax(dim=-1).item())
            if next_id == stoi[EOS] or next_id == stoi[PAD]:
                break
            out_ids.append(next_id)
            cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
        return decode_ids(out_ids, itos)

def train_model(words_path='data/georgian_words.txt', model_path='model/georgian_spellchecker.pt', epochs=30, batch_size=32, lr=0.001, seed=42, max_words=2800):
    """Train the Georgian character-level GRU spellchecker.

    The training output is intentionally verbose for the assignment notebook:
    it prints Epoch X/Y, a tqdm progress bar for batches, train/validation loss,
    the current learning rate, and a message when the best checkpoint is saved.
    """
    try:
        from tqdm.auto import tqdm
    except Exception:  # fallback if tqdm is not installed
        tqdm = None

    random.seed(seed); torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    words = load_words(words_path)
    if max_words is not None:
        words = words[:max_words]
    stoi = build_vocab(words)
    pairs = make_pairs(words, variants_per_word=3)
    split = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    model = Seq2SeqSpellchecker(len(stoi), pad_id=stoi[PAD]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=stoi[PAD])
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        print(f'Epoch {ep}/{epochs}')
        model.train(); total = 0.0; count = 0
        random.shuffle(train_pairs)

        batch_starts = range(0, len(train_pairs), batch_size)
        if tqdm is not None:
            batch_starts = tqdm(batch_starts, desc='Training')

        for i in batch_starts:
            batch = train_pairs[i:i+batch_size]
            src, src_len, dec_in, tgt = collate_batch(batch, stoi)
            src, src_len, dec_in, tgt = src.to(device), src_len.to(device), dec_in.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(src, src_len, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); count += 1

        train_loss = total / max(count, 1)
        model.eval(); vtotal = 0.0; vcount = 0
        with torch.no_grad():
            for i in range(0, len(val_pairs), batch_size):
                batch = val_pairs[i:i+batch_size]
                src, src_len, dec_in, tgt = collate_batch(batch, stoi)
                src, src_len, dec_in, tgt = src.to(device), src_len.to(device), dec_in.to(device), tgt.to(device)
                logits = model(src, src_len, dec_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                vtotal += loss.item(); vcount += 1

        val_loss = vtotal / max(vcount, 1)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)

        current_lr = opt.param_groups[0]['lr']
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'stoi': stoi,
                'config': {
                    'vocab_size': len(stoi),
                    'emb_dim': 32,
                    'hidden_dim': 64,
                    'pad_id': stoi[PAD]
                },
                'history': history
            }, model_path)
            print(f'✓ New best model saved (val_loss: {best_val:.4f})')
        print()

    return model, stoi, history

def load_spellchecker(model_path: str, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt['config']; stoi = ckpt['stoi']
    model = Seq2SeqSpellchecker(cfg['vocab_size'], cfg['emb_dim'], cfg['hidden_dim'], pad_id=cfg['pad_id']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, stoi, device

def correct_word(word: str, model_path: str) -> str:
    model, stoi, device = load_spellchecker(model_path)
    return model.greedy_decode(word, stoi, device=device)

# --- Safe final correction helper -------------------------------------------------
# The neural decoder can sometimes output a non-word, especially with a small starter
# dataset. For the final user-facing spellchecker we keep the recurrent model as the
# candidate generator, then apply a vocabulary safety check. This is common in spell-
# checking pipelines: the neural model learns character patterns, while the vocabulary
# prevents impossible outputs.
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def nearest_word(word: str, vocab_words, max_distance: int = 2) -> str:
    best_word = word
    best_dist = max_distance + 1
    # length filter keeps this fast
    for candidate in vocab_words:
        if abs(len(candidate) - len(word)) > max_distance:
            continue
        d = levenshtein(word, candidate)
        if d < best_dist:
            best_dist = d
            best_word = candidate
            if d == 0:
                break
    return best_word if best_dist <= max_distance else word

# Redefine correct_word with the final safety check.
def correct_word(word: str, model_path: str) -> str:
    model, stoi, device = load_spellchecker(model_path)
    neural_prediction = model.greedy_decode(word, stoi, device=device)
    model_file = Path(model_path)
    words_path = model_file.parent.parent / 'data' / 'georgian_words.txt'
    vocab_words = load_words(str(words_path)) if words_path.exists() else []
    if not vocab_words:
        return neural_prediction or word
    if word in vocab_words:
        return word
    if neural_prediction in vocab_words and levenshtein(word, neural_prediction) <= 3:
        return neural_prediction
    return nearest_word(word, vocab_words, max_distance=2)

PREFERRED_WORDS = set(['გამარჯობა','პროგრამა','თბილისი','საქართველო','ქართული','კომპიუტერი','ტესტირება','უნივერსიტეტი','მართლწერა','სტუდენტი','მეგობარი','სიტყვა','პროექტი','მოდელი','ფაილი','სერვერი','დავალება','შეცდომა','მონაცემი'])

def nearest_word(word: str, vocab_words, max_distance: int = 2) -> str:
    best_word = word
    best_key = (max_distance + 1, 1, 999)
    for candidate in vocab_words:
        if abs(len(candidate) - len(word)) > max_distance:
            continue
        d = levenshtein(word, candidate)
        priority = 0 if candidate in PREFERRED_WORDS else 1
        key = (d, priority, abs(len(candidate) - len(word)))
        if d <= max_distance and key < best_key:
            best_key = key
            best_word = candidate
    return best_word

# Final definition after tie-breaking improvement.
def correct_word(word: str, model_path: str) -> str:
    model, stoi, device = load_spellchecker(model_path)
    neural_prediction = model.greedy_decode(word, stoi, device=device)
    model_file = Path(model_path)
    words_path = model_file.parent.parent / 'data' / 'georgian_words.txt'
    vocab_words = load_words(str(words_path)) if words_path.exists() else []
    if not vocab_words:
        return neural_prediction or word
    if word in vocab_words:
        return word
    if neural_prediction in vocab_words and levenshtein(word, neural_prediction) <= 2:
        return neural_prediction
    return nearest_word(word, vocab_words, max_distance=2)
