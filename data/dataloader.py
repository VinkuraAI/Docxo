import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataloader(config, rank, world_size):
    # Load the dataset
    dataset = load_dataset(config['dataset']['name'], config['dataset']['subset'], 
                          split=config['dataset']['split'])
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], max_length=config['model']['max_length'], 
                        padding='max_length', truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    # Set format to PyTorch tensors
    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    
    # Use DistributedSampler for DDP
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        drop_last=True  # Avoid issues with uneven batches
    )
    
    return dataloader