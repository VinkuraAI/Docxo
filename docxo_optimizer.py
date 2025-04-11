import torch
import torch.distributed as dist

def custom_training_step(model, optimizer, batch, rank, local_steps, step_count):
    # Move batch to GPU
    input_ids = batch['input_ids'].to(rank)
    attention_mask = batch['attention_mask'].to(rank)
    labels = input_ids.clone() 
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Local update
    optimizer.step()
    optimizer.zero_grad()
    
    # Synchronize parameters every 'local_steps' steps
    if step_count % local_steps == 0:
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()
    
    return loss.item()