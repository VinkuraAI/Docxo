import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AdamW
import yaml
from data.dataloader import get_dataloader
from docxo_optimizer import custom_training_step
import matplotlib.pyplot as plt

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def setup(rank, world_size):
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'  # Update to GCP master IP if multi-node
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    # Clean up distributed environment
    dist.destroy_process_group()

def train(rank, world_size):
    # Set up distributed training
    setup(rank, world_size)
    
    # Create directories (only on rank 0)
    if rank == 0:
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
        os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Load data
    dataloader = get_dataloader(config, rank, world_size)
    
    # Training loop
    model.train()
    total_steps = 0
    losses = []
    
    for epoch in range(config['training']['num_epochs']):
        dataloader.sampler.set_epoch(epoch)  # Ensure shuffling across epochs
        for batch in dataloader:
            total_steps += 1
            
            # Custom training step
            loss = custom_training_step(
                model, optimizer, batch, rank,
                config['training']['local_steps'], total_steps
            )
            losses.append(loss)
            
            # Log progress (rank 0 only)
            if rank == 0 and total_steps % 10 == 0:
                print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}, "
                      f"Step {total_steps}/{config['training']['max_steps']}, "
                      f"Loss: {loss:.4f}")
            
            # Stop after max_steps
            if total_steps >= config['training']['max_steps']:
                break
        
        if total_steps >= config['training']['max_steps']:
            break
    
    # Save model and plot (rank 0 only)
    if rank == 0:
        model.module.save_pretrained(config['logging']['checkpoint_dir'])
        
        # Plot loss
        plt.style.use('dark_background')
        plt.plot(losses, color='#00d4ff', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Vinkura DOCXO Training Loss')
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.savefig(os.path.join(config['logging']['log_dir'], 'loss.png'))
        plt.close()
    
    # Clean up
    cleanup()

if __name__ == "__main__":
    # Launch training
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 1)  # Single GPU fallback