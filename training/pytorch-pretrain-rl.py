import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# ------------------------------
# Toy vocabulary and enums
# ------------------------------
vocab = {"<pad>":0, "<unk>":1, "mozilla":2, "chrome":3, "mobile":4, "desktop":5, "dark":6, "light":7}
vocab_size = len(vocab)
max_len = 5
enum_sizes = [2, 3]  # e.g., Dark/Light, Device type

# ------------------------------
# Helper: sample sequence from logits
# ------------------------------
def sample_seq(logits):
    seq = []
    for t in range(logits.size(1)):
        dist = torch.distributions.Categorical(logits=logits[:,t,:])
        token = dist.sample()
        seq.append(token)
    return torch.stack(seq, dim=1)

# ------------------------------
# Policy Model: Seq2Seq + Enums
# ------------------------------
class Website2ConfigPolicy(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, enum_sizes=[2,3]):
        super().__init__()
        self.web_embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # UA decoder
        self.ua_embed = nn.Embedding(vocab_size, hidden_dim)
        self.ua_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ua_out = nn.Linear(hidden_dim, vocab_size)

        # FP decoder
        self.fp_embed = nn.Embedding(vocab_size, hidden_dim)
        self.fp_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fp_out = nn.Linear(hidden_dim, vocab_size)

        # Enum heads
        self.enum_heads = nn.ModuleList([nn.Linear(hidden_dim, n) for n in enum_sizes])

    def forward(self, url_tokens, ua_input=None, fp_input=None):
        # Encode website
        web_emb = self.web_embed(url_tokens)
        _, h = self.encoder(web_emb)
        ctx = h  # (1, B, H)

        B = url_tokens.size(0)

        # UA decoding
        ua_seq = ua_input if ua_input is not None else torch.zeros((B, max_len), dtype=torch.long)
        ua_emb = self.ua_embed(ua_seq)
        ua_out, _ = self.ua_decoder(ua_emb, ctx)
        ua_logits = self.ua_out(ua_out)

        # FP decoding
        fp_seq = fp_input if fp_input is not None else torch.zeros((B, max_len), dtype=torch.long)
        fp_emb = self.fp_embed(fp_seq)
        fp_out, _ = self.fp_decoder(fp_emb, ctx)
        fp_logits = self.fp_out(fp_out)

        # Enum logits
        enum_logits = [head(ctx.squeeze(0)) for head in self.enum_heads]

        return ua_logits, fp_logits, enum_logits

# ------------------------------
# Toy reward function
# ------------------------------
def reward_fn(ua_seq, enums):
    # Reward=1 if UA contains "mobile" token and first enum = 1
    return 1.0 if (4 in ua_seq and enums[0] == 1) else 0.0

# ------------------------------
# Generate toy pretraining data
# ------------------------------
def generate_toy_data(batch_size=16):
    url_batch = []
    ua_batch = []
    fp_batch = []
    enum_batch = []
    for _ in range(batch_size):
        url_tokens = [random.choice(list(vocab.values())) for _ in range(5)]
        ua_tokens = [random.choice(list(vocab.values())) for _ in range(max_len)]
        fp_tokens = [random.choice(list(vocab.values())) for _ in range(max_len)]
        enums = [random.randint(0, enum_sizes[0]-1), random.randint(0, enum_sizes[1]-1)]

        url_batch.append(url_tokens)
        ua_batch.append(ua_tokens)
        fp_batch.append(fp_tokens)
        enum_batch.append(enums)

    return (torch.tensor(url_batch), torch.tensor(ua_batch), torch.tensor(fp_batch),
            [torch.tensor([e[i] for e in enum_batch]) for i in range(len(enum_sizes))])

# ------------------------------
# Initialize model & optimizer
# ------------------------------
policy = Website2ConfigPolicy(vocab_size=vocab_size, enum_sizes=enum_sizes)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

# ------------------------------
# Step 1: Pretraining with Teacher Forcing
# ------------------------------
pretrain_steps = 100
for step in range(pretrain_steps):
    url_batch, ua_batch, fp_batch, enum_batch = generate_toy_data()

    ua_logits, fp_logits, enum_logits = policy(url_batch, ua_input=ua_batch, fp_input=fp_batch)

    # Token sequence loss
    ua_loss = F.cross_entropy(ua_logits.view(-1, vocab_size), ua_batch.view(-1))
    fp_loss = F.cross_entropy(fp_logits.view(-1, vocab_size), fp_batch.view(-1))

    # Enum loss
    enum_loss = sum(F.cross_entropy(logits, target) for logits, target in zip(enum_logits, enum_batch))

    loss = ua_loss + fp_loss + enum_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Pretrain Step {step}, Loss={loss.item():.4f}")

# ------------------------------
# Step 2: RL Fine-tuning
# ------------------------------
rl_steps = 100
for step in range(rl_steps):
    url_batch, _, _, _ = generate_toy_data(batch_size=4)  # RL batch

    ua_logits, fp_logits, enum_logits = policy(url_batch)

    ua_seq = sample_seq(ua_logits)
    fp_seq = sample_seq(fp_logits)

    enum_actions = []
    enum_dists = []
    for logits in enum_logits:
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        enum_actions.append(act)
        enum_dists.append(dist)

    # Compute reward (per example)
    rewards = []
    for i in range(url_batch.size(0)):
        r = reward_fn(ua_seq[i].tolist(), [e[i].item() for e in enum_actions])
        rewards.append(r)
    rewards = torch.tensor(rewards, dtype=torch.float)

    # Compute log probs
    log_probs = 0
    for t in range(max_len):
        log_probs += torch.distributions.Categorical(logits=ua_logits[:,t,:]).log_prob(ua_seq[:,t])
        log_probs += torch.distributions.Categorical(logits=fp_logits[:,t,:]).log_prob(fp_seq[:,t])
    log_probs += sum(d.log_prob(a) for d,a in zip(enum_dists, enum_actions))

    loss = -(log_probs * rewards.mean())  # REINFORCE

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"RL Step {step}, Avg Reward={rewards.mean().item():.2f}, UA Example={ua_seq[0].tolist()}, Enums={[e[0].item() for e in enum_actions]}")
