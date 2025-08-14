import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from evo.tokenisation import Tokenizer
from evo.model import StripedHyena
from evo.utils import dotdict
from evo.modeling_hyena import StripedHyenaModelForCausalLM
from evo.configuration_hyena import StripedHyenaConfig
import pandas as pd
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

class HyenaSeq2SeqDataset(Dataset):
    def __init__(self, aa_nt_df, tokenizer, max_seq_length=100, device='cuda'):
        self.aa_nt_df = aa_nt_df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device

    def __len__(self):
        return len(self.aa_nt_df)

    def __getitem__(self, idx):
        input_ids, attention_mask, labels = self.tokenizer.format_dataset_torch(
            self.aa_nt_df.iloc[[idx]], max_seq_length=self.max_seq_length, device=self.device
        )
        return input_ids.squeeze(0), attention_mask.squeeze(0), labels.squeeze(0)

class HyenaDataLoader:
    def __init__(self, csv_path, tokenizer, max_seq_length=100, batch_size=2, device='cuda', train_size=0.8, val_size=0.1, test_size=0.1):
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
        self.val_dataset = HyenaSeq2SeqDataset(val_df, self.tokenizer, self.max_seq_length, self.device)
        self.test_dataset = HyenaSeq2SeqDataset(test_df, self.tokenizer, self.max_seq_length, self.device)

    def get_loaders(self, rank):
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=torch.distributed.get_world_size(), rank=rank)
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=torch.distributed.get_world_size(), rank=rank)
        test_sampler = DistributedSampler(self.test_dataset, num_replicas=torch.distributed.get_world_size(), rank=rank)
        train_loader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        val_loader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler, micro_batch_size=1, accumulation_steps=2):
    model.train()
    total_loss = 0.0
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    optimizer.zero_grad()
    for step, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.scale(optimizer)
            torch.cuda.empty_cache()
            scaler.update
            optimizer.zero_grad()
            scheduler.step()
            torch.cuda.empty_cache()

        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()
        # scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for step, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, os.path.join(output_dir, f"checkpoint-epoch{epoch}.pt"))

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_model(model, tokenizer, dataloader_provider, epochs=5, lr=5e-5, output_dir="checkpoints", warmup_steps=50, device="cuda", rank=0, micro_batch_size=1, accumulation_steps=2):
    train_loader, val_loader, test_loader = dataloader_provider.get_loaders(rank)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, total_iters=warmup_steps)
    scaler = GradScaler()
    for epoch in range(1, epochs + 1):
        print(f"\n🚀 Epoch {epoch}/{epochs} on rank {rank}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, scaler)
        print(f"🟢 Train Loss: {train_loss:.4f}")
        val_loss = evaluate(model, val_loader, device)
        print(f"🔵 Validation Loss: {val_loss:.4f}")
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, output_dir)
    test_loss = evaluate(model, test_loader, device)
    print(f"\n🔶 Final Test Loss (Rank {rank}): {test_loss:.4f}")

def run(args):
    rank, local_rank, world_size = setup()
    tokenizer = Tokenizer()
    dataloader_provider = HyenaDataLoader(args["csv_path"], tokenizer, max_seq_length=100, device=f"cuda:{local_rank}")
    model = StripedHyenaModelForCausalLM(args["config"])
    model.gradient_checkpointing_enable()  
    ddp_model = DDP(model.to(local_rank), device_ids=[local_rank])
    train_model(ddp_model, tokenizer, dataloader_provider,
                epochs=args["epochs"],
                output_dir=args["output_dir"],
                device=f"cuda:{local_rank}",
                rank=rank)
    cleanup()

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     with open("/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/config.json") as f:
#         config_dict = json.load(f)
#     args = {
#         "csv_path": "/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/data/E_coli_genomes_aa_nt_seqs.csv",
#         "config": StripedHyenaConfig(**config_dict),
#         "epochs": 10,
#         "output_dir": "hyena_checkpoints"
#     }
#     mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    with open("/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/config.json") as f:
        config_dict = json.load(f)
    args = {
        "csv_path": "/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/data/E_coli_genomes_aa_nt_seqs.csv",
        "config": StripedHyenaConfig(**config_dict),
        "epochs": 10,
        "output_dir": "hyena_checkpoints"
    }
    run(args)
