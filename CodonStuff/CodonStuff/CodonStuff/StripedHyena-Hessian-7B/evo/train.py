# train.py (modified to use a causal loss wrapper around StripedHyena)
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .auxiliary_file import train_validate_test_split
from sklearn.model_selection import train_test_split
from evo.tokenisation import Tokenizer
from evo.model import StripedHyena
from torch.utils.data import DataLoader, TensorDataset
from evo.utils import dotdict, print_rank_0
import yaml
from evo.modeling_hyena import StripedHyenaPreTrainedModel , StripedHyenaModelForCausalLM
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
import json
from torch.utils.data import Dataset
import pandas as pd
from evo.configuration_hyena import StripedHyenaConfig
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
import Levenshtein
import numpy as np
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from types import SimpleNamespace

def split_codons_from_nt_string(nt_str):
    """Return a list of codon strings, ignore trailing incomplete codon."""
    codons = [nt_str[i:i+3] for i in range(0, len(nt_str), 3)]
    codons = [c for c in codons if len(c) == 3]
    return codons

def codon_list_levenshtein(a, b):
    """Levenshtein distance between two lists of codons (returns integer)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

# ---------------------------
# Collate (pads + builds mask)
# ---------------------------
def hyena_collate_fn(batch, pad_token_id=0):
    """
    batch: list of tuples (input_ids_1D, labels_1D)
    Returns: input_ids[B, L], attention_mask[B, L], labels[B, L]
    """
    input_seqs, label_seqs = zip(*batch)  # exactly two items per sample

    input_tensors = [torch.as_tensor(x, dtype=torch.long) for x in input_seqs]
    label_tensors = [torch.as_tensor(y, dtype=torch.long) for y in label_seqs]

    input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=pad_token_id)
    labels    = pad_sequence(label_tensors, batch_first=True, padding_value=-100)

    attention_mask = (input_ids != pad_token_id).long()  # 1=keep, 0=pad
    return input_ids, attention_mask, labels


# ---------------------------
# Dataset
# ---------------------------
class HyenaSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, aa_nt_df, tokenizer, max_seq_length=2048, device='cuda'):
        self.aa_nt_df = aa_nt_df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device  # not used inside dataset; move-to-device in training loop

    def __len__(self):
        return len(self.aa_nt_df)

    def __getitem__(self, idx):
        # Build a SINGLE sample (1D lists), no padding, no mask
        aa = self.aa_nt_df.iloc[idx, 0]
        nt = self.aa_nt_df.iloc[idx, 1]

        # Make the concatenated token sequence: <START> AAs <SEP> NTs <END>
        # and the label sequence: -100 for <= SEP, codon ids after
        tokens_2d = self.tokenizer.tokenise_aa_nt_pair([aa], [nt])  # returns shape (1, L) torch.LongTensor
        token_ids = tokens_2d[0].tolist()  # 1D list

        # Build labels aligned to token_ids
        if self.tokenizer.sep_token_id not in token_ids:
            # malformed pair; you can raise or skip. Here we raise to surface data issues.
            raise ValueError("No <SEP> in tokenized pair")

        sep_idx = token_ids.index(self.tokenizer.sep_token_id)
        labels = [(-100 if i <= sep_idx else tid) for i, tid in enumerate(token_ids)]

        # (Optional) sanity: labels within vocab except masked
        # Use tokenizer.vocab_size (not a global config)
        if hasattr(self.tokenizer, "vocab_size"):
            for t in labels:
                if t != -100 and not (0 <= t < self.tokenizer.vocab_size):
                    raise ValueError(f"Label out of range: {t}")

        # Return ONLY (input_ids_1D, labels_1D); collate will pad & make mask
        return token_ids, labels


# ---------------------------
# DataLoader wrapper
# ---------------------------
class HyenaDataLoader:
    def __init__(self, csv_path, tokenizer, max_seq_length=2048, batch_size=32,
                 device='cuda', train_size=0.8, val_size=0.1, test_size=0.1):
        assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "Splits must sum to 1."

        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        self._prepare_datasets()

    def _prepare_datasets(self):
        df = pd.read_csv(self.csv_path, header=None)
        df.columns = ['aa_seq', 'nt_seq']

        train_df, temp_df = train_test_split(df, test_size=(1.0 - self.train_size), random_state=42)
        val_split = self.val_size / (self.val_size + self.test_size)
        val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_split), random_state=42)

        self.train_dataset = HyenaSeq2SeqDataset(train_df, self.tokenizer, self.max_seq_length, self.device)
        self.val_dataset   = HyenaSeq2SeqDataset(val_df,   self.tokenizer, self.max_seq_length, self.device)
        self.test_dataset  = HyenaSeq2SeqDataset(test_df,  self.tokenizer, self.max_seq_length, self.device)

    def get_loaders(self):
        pad_id = self.tokenizer.pad_token_id
        collate = lambda b: hyena_collate_fn(b, pad_token_id=pad_id)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,  collate_fn=collate)
        val_loader   = DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False, collate_fn=collate)
        test_loader  = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False, collate_fn=collate)
        return train_loader, val_loader, test_loader

# ---------------------------
# Causal Loss Wrapper
# ---------------------------
class StripedHyenaCausalLossWrapper(nn.Module):
    """
    Wraps the original StripedHyena model (unchanged) and provides:
    - attention_mask handling (pad token -> mask)
    - causal (shifted) cross-entropy loss with ignore_index (-100)
    - HF-like return object: outputs.loss and outputs.logits (SimpleNamespace)
    """
    def __init__(self, base_model, pad_token_id, ignore_index=-100, label_smoothing=0.1):
        super().__init__()
        self.base_model = base_model
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            label_smoothing=label_smoothing
        )

    def forward(self, input_ids, attention_mask=None, labels=None, inference_params_dict=None):
        # Build attention_mask if not provided: 1 for non-pad, 0 for pad
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Move masks to device and ensure same device as input_ids
        attention_mask = attention_mask.to(input_ids.device)

        # Call original StripedHyena: note signature forward(x, inference_params_dict=None, padding_mask=None)
        # We pass padding_mask=attention_mask (shape: [B, L]) — model will expand dim internally where needed.
        logits, inference_out = self.base_model(x=input_ids, inference_params_dict=inference_params_dict, padding_mask=attention_mask)

        # logits: (B, L, V)
        if labels is not None:
            labels = labels.to(input_ids.device)
            # Shift logits and labels for causal LM loss (predict token t from tokens <= t-1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return SimpleNamespace(loss=loss, logits=logits, inference_out=inference_out)
        else:
            return SimpleNamespace(logits=logits, inference_out=inference_out)

# ---------------------------
# Training / Evaluation functions (updated to use wrapper)
# ---------------------------
def train_one_epoch(model, loss_wrapper, dataloader, optimizer, scheduler, device):
    """
    model: base StripedHyena model (parameters live here)
    loss_wrapper: StripedHyenaCausalLossWrapper instance (calls base model, computes loss)
    """
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        try:
            input_ids, attention_mask, labels = batch
        except ValueError as e:
            # Skip malformed batch
            print(f"⚠️ Skipping batch {step} due to ValueError: {e}")
            continue

        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Get loss and logits from wrapper
        outputs = loss_wrapper(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients on base model parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('nan')
    return avg_loss

@torch.no_grad()
def evaluate(model, loss_wrapper, dataloader, tokenizer, device):
    model.eval()
    loss_wrapper.eval()
    total_loss = 0.0

    total_codon_lev = 0
    total_codon_count_for_norm = 0  # used to normalize Levenshtein by codon count
    total_correct_codons = 0
    total_codons_compared = 0

    for step, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # loss via wrapper
        outputs = loss_wrapper(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.detach().item()

        # Reconstruct AA sequences (strings) from input_ids up to SEP
        aa_input_ids = []
        for seq in input_ids:
            sep_index = (seq == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_index) > 0:
                aa_input_ids.append(seq[:sep_index[0]].tolist())
            else:
                aa_input_ids.append(seq.tolist())  # fallback: use whole seq

        aa_seqs = tokenizer._detokenise_aa_seqs(aa_input_ids)

        # Ground-truth nucleotide sequences (strings)
        nt_label_ids = []
        for label_seq in labels:
            seq_ids = label_seq[label_seq != -100].tolist()
            nt_label_ids.append(seq_ids)

        true_nts = tokenizer._detokenise_nt_seqs(nt_label_ids)  # returns list of nucleotide strings

        # Use the wrapper for generation (it accepts attention_mask)
        pred_nts = tokenizer.translate_aa_into_nt_torch(
            loss_wrapper, aa_seqs, max_seq_length=2048,
            return_string=True, batch_size=32, device=device, temperature=0.5
        )

        # compute codon-level metrics
        for pred_nt, true_nt in zip(pred_nts, true_nts):
            # skip empty references
            if not true_nt:
                continue

            # split into codon lists
            pred_codons = split_codons_from_nt_string(pred_nt)
            true_codons = split_codons_from_nt_string(true_nt)

            # after we have pred_codons and true_codons:
            print("AA:", aa_seqs)  # show amino acid sequence used for generation
            print("PRED codons:", pred_codons[:10])
            print("TRUE codons:", true_codons[:10])
            break  # debug only first sample

            # codon-level Levenshtein
            dist_codons = codon_list_levenshtein(pred_codons, true_codons)
            norm_denom = max(len(pred_codons), len(true_codons), 1)
            total_codon_lev += dist_codons
            total_codon_count_for_norm += norm_denom

            # codon accuracy (position-wise up to min length)
            min_len = min(len(pred_codons), len(true_codons))
            if min_len > 0:
                for i in range(min_len):
                    if pred_codons[i] == true_codons[i]:
                        total_correct_codons += 1
                total_codons_compared += min_len

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('nan')
    # normalised Levenshtein (codon-level)
    avg_norm_lev = (total_codon_lev / total_codon_count_for_norm) if total_codon_count_for_norm > 0 else float('inf')
    codon_accuracy = (total_correct_codons / total_codons_compared) if total_codons_compared > 0 else 0.0

    print(f"📉 Avg Loss: {avg_loss:.4f}")
    print(f"🧬 Avg Normalised Codon Levenshtein Distance: {avg_norm_lev:.4f}")
    print(f"🎯 Codon Accuracy: {codon_accuracy:.4f}")

    return avg_loss, avg_norm_lev, codon_accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint-epoch{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")

def safe_plot(
    values,
    label,
    title,
    xlabel,
    ylabel,
    filename,
    output_dir,
    color=None,
    marker='o'
):
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(values) + 1), values, label=label, color=color, marker=marker)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    except Exception as e:
        print(f"⚠️ Plotting failed for {filename}: {e}")

def train_model(
    model,
    tokenizer,
    dataloader_provider,  # Instance of HyenaDataLoader
    epochs=100,
    lr=1e-4,
    weight_decay=0.01,
    output_dir="checkpoints",
    max_seq_length=2048,
    warmup_steps=100,
    device="cuda"
):
    train_loader, val_loader, test_loader = dataloader_provider.get_loaders()

    model.to(device)

    # Create the loss wrapper (wraps the base model without changing it)
    loss_wrapper = StripedHyenaCausalLossWrapper(base_model=model, pad_token_id=tokenizer.pad_token_id, ignore_index=-100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    # 🧮 Track losses for plotting
    train_losses = []
    val_losses = []
    val_levenshteins = []
    val_perplexities = []
    codon_accuracies = []

    for epoch in range(1, epochs + 1):
        print(f"\n🚀 Epoch {epoch}/{epochs}")

        train_loss = train_one_epoch(model, loss_wrapper, train_loader, optimizer, scheduler, device)
        print(f"🟢 Train Loss: {train_loss:.4f}")

        if epoch % 25 == 0: #or epoch == 1:
            # full evaluation every few epochs
            val_loss, val_lev, val_acc = evaluate(model, loss_wrapper, val_loader, tokenizer, device)
            val_perplexity = np.exp(val_loss) if val_loss < 100 else float('inf')
            print(f"🎯 Codon Accuracy: {val_acc:.4f}")
            print(f"📉 Perplexity: {val_perplexity:.2f}")
            print(f"🔵 Validation Loss: {val_loss:.4f}")
            print(f"🧬 Validation Levenshtein Distance: {val_lev:.4f}")

            # record for plotting
            val_losses.append(val_loss)
            val_levenshteins.append(val_lev)
            val_perplexities.append(val_perplexity)
            codon_accuracies.append(val_acc)
        else:
            # cheap “loss‐only” evaluation
            val_loss = evaluate_loss_only(model, loss_wrapper, val_loader, device)
            print(f"[Eval-lite] Loss: {val_loss:.4f}")
            val_losses.append(val_loss)

        train_losses.append(train_loss)

        if epoch == 100:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, output_dir)

    # plotting (same as before)
    safe_plot(
        values=train_losses,
        label="Train Loss",
        title="Train Loss Curve",
        xlabel="Epoch",
        ylabel="Loss",
        filename="train_loss_curve.png",
        output_dir=output_dir,
        color="blue",
        marker="o"
    )
    safe_plot(
        values=val_losses,
        label="Val Loss",
        title="Val Loss Curve",
        xlabel="Epoch",
        ylabel="Loss",
        filename="val_loss_curve.png",
        output_dir=output_dir,
        color="red",
        marker="s"
    )
    safe_plot(
        values=val_levenshteins,
        label="Val Levenshtein",
        title="Validation Levenshtein Curve",
        xlabel="Epoch",
        ylabel="Normalised Levenshtein Distance",
        filename="val_levenshtein_curve.png",
        output_dir=output_dir,
        color="purple",
        marker="^"
    )
    safe_plot(
        values=val_perplexities,
        label="Val Perplexity",
        title="Validation Perplexity",
        xlabel="Epoch",
        ylabel="Perplexity",
        filename="val_perplexity_curve.png",
        output_dir=output_dir,
        color="darkorange",
        marker="d"
    )
    safe_plot(
        values=codon_accuracies,
        label="Codon Accuracy",
        title="Validation Codon Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        filename="val_codon_accuracy_curve.png",
        output_dir=output_dir,
        color="green",
        marker="x"
    )

    # df = pd.DataFrame({
    #     "epoch": list(range(1, epochs+1)),
    #     "train_loss": train_losses,
    #     "val_loss": val_losses,
    #     "val_levenshtein": val_levenshteins,
    #     "val_perplexity": val_perplexities,
    #     "val_codon_accuracy": codon_accuracies
    # })

    # df.to_csv(os.path.join(output_dir, "losses.csv"), index=False)

    test_loss, test_lev, test_acc = evaluate(model, loss_wrapper, test_loader, tokenizer, device)
    print(f"\n🔶 Final Test Loss: {test_loss:.4f}")
    print(f"🧬 Final Test Levenshtein Distance: {test_lev:.4f}")
    print(f"🎯 Final Test Codon Accuracy: {test_acc:.4f}")

@torch.no_grad()
def evaluate_loss_only(model, loss_wrapper, dataloader, device="cuda"):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = loss_wrapper(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else float('nan')

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Load config
    with open("/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/config.json") as f:
        config_dict = json.load(f)

    config = StripedHyenaConfig(**config_dict)
    print_rank_0(f"Config: {config}")

    tokenizer = Tokenizer()
    # Ensure model vocab matches tokenizer (handles non-contiguous id spaces)
    config.vocab_size = tokenizer.vocab_size
    dataloader_provider = HyenaDataLoader("/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/data/E_coli_genomes_aa_nt_seqs.csv", tokenizer, max_seq_length=2048)
    #dataloader_provider = HyenaDataLoader("/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/data/test_seq.csv", tokenizer, max_seq_length=2048)

    # Base model (unchanged)
    model = StripedHyena(config)
    # --- Sanity-check: run one small generation check on val set (debug) ---
    # Insert this AFTER:
    #   dataloader_provider = HyenaDataLoader(...)
    #   model = StripedHyena(config)
    #   loss_wrapper = StripedHyenaCausalLossWrapper(base_model=model, pad_token_id=tokenizer.pad_token_id, ignore_index=-100)
    #
    # and BEFORE:
    #   train_model(...)
    #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loss_wrapper = StripedHyenaCausalLossWrapper(base_model=model, pad_token_id=tokenizer.pad_token_id, ignore_index=-100)
    loss_wrapper.to(device)
    loss_wrapper.eval()
    model.eval()

    train_loader, val_loader, test_loader = dataloader_provider.get_loaders()

    # print("\n🔍 Running quick sanity checks on the validation loader (1 batch)...")
    # with torch.no_grad():
    #     # use an iterator so next(...) gets the next batch each loop
    #     val_iter = iter(val_loader)
    #     for i in range(10):
    #         try:
    #             input_ids_batch, attention_mask_batch, labels_batch = next(val_iter)
    #         except StopIteration:
    #             print("Reached end of val loader")
    #             break

    #         # Move first sample to device for decoding/generation
    #         input_ids = input_ids_batch[0].to(device)
    #         attention_mask = attention_mask_batch[0].to(device)
    #         labels = labels_batch[0].to(device)

    #         # Report how many tokens are not -100 in labels
    #         non_100_count = (labels != -100).sum().item()
    #         print(f"\nSample {i}: labels non-100 count (nucleotide tokens) = {non_100_count}")

    #         # Decode ground truth nucleotide string from labels (strip -100)
    #         nt_ids = labels[labels != -100].tolist()
    #         if len(nt_ids) == 0:
    #             print("  ⚠️ Ground-truth label is empty after masking (all -100).")
    #         else:
    #             true_nt = tokenizer._detokenise_nt_seqs([nt_ids])[0]
    #             print("  label nt sample (first 200 bp):", true_nt[:200])

    #         # Reconstruct AA input up to <SEP>
    #         sep_positions = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
    #         if len(sep_positions) > 0:
    #             sep_idx = sep_positions[0].item()
    #             aa_ids = input_ids[:sep_idx].cpu().tolist()
    #         else:
    #             aa_ids = input_ids.cpu().tolist()
    #         aa_seq = tokenizer._detokenise_aa_seqs([aa_ids])[0]
    #         print("  aa (input) sample:", aa_seq)
    #         nt_ids = labels[labels != -100].tolist()
    #         max_out_codon_len = len(nt_ids) if len(nt_ids) > 0 else 128  # sensible cap
    #         # Generate prediction from AA using the wrapper (handles attention_mask)
    #         pred_nt = tokenizer.translate_aa_into_nt_torch(
    #             loss_wrapper, [aa_seq], max_seq_length = max_out_codon_len, return_string=True, batch_size=1, device=device, temperature=0.8
    #         )[0]
    #         print("  pred nt sample (first 200 bp):", pred_nt[:200])

    #         # Quick codon-level split & comparison (first min codons)
    #         def split_codons(s): return [s[i:i+3] for i in range(0, len(s), 3) if len(s[i:i+3])==3]
    #         pred_codons = split_codons(pred_nt)
    #         true_codons = split_codons(true_nt) if len(nt_ids) > 0 else []
    #         min_len = min(len(pred_codons), len(true_codons))
    #         if min_len > 0:
    #             matches = sum(1 for a,b in zip(pred_codons[:min_len], true_codons[:min_len]) if a==b)
    #             print(f"  codon matches (first {min_len} codons): {matches}/{min_len} = {matches/min_len:.3f}")
    #         else:
    #             print("  No codons to compare for this sample.")

    #         # stop after first sample if you're only debugging one
    #         break

    # print("🔍 Sanity checks complete — remove or comment this block before long training runs.\n")

    # Train
    train_model(model, tokenizer, dataloader_provider, epochs=100, output_dir="hyena_checkpoints")
