import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, default_data_collator
from tqdm import tqdm

def train_model(model, train_dataloader, eval_dataloader, num_epochs, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        train_epoch_loss = total_loss / len(train_dataloader)
        print(f"{epoch=}: {train_epoch_loss=} {eval_epoch_loss=}")
    
    return model