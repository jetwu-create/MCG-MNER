from tqdm import tqdm
import torch
import numpy as np
from utils.evaluate_utils import evaluate, update_best_checkpoint


def train(
    epochs,
    model,
    tokenizer,
    train_dataloader,
    test_dataloader,
    optimizer,
    writer,
    device,
    eval_every_n_batches,
    pred_every_n_batches,
    generation_kwargs,
    options,
    metric_name_to_choose_best,
    metric_avg_to_choose_best,
):
    
    metrics_best = {}
    for epoch in range(epochs):
        
        print(f"Epoch [{epoch + 1} / {epochs}]\n")
        metrics_best = train_epoch(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device,
            epoch=epoch,
            metrics_best=metrics_best,
        )
        
        evaluate_metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=test_dataloader,
                    writer=writer,
                    device=device,
                    epoch=epoch,
                    generation_kwargs=generation_kwargs,
                    options=options,
                )
        metrics_best = update_best_checkpoint(
                        metrics_best=metrics_best,
                        metrics_new=evaluate_metrics,
                        metric_name=metric_name_to_choose_best,
                        metric_avg=metric_avg_to_choose_best,
                        model=model,
                        tokenizer=tokenizer,
                )


def train_epoch(
    model,
    tokenizer,
    train_dataloader,
    optimizer,
    writer,
    device,
    epoch,
    metrics_best,
):
    epoch_loss = []

    for i, inputs in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
    ):
        
        model.train()
        optimizer.zero_grad()
        inputs.pop('instances')
        answers = inputs.pop("answers")
        answers = torch.tensor(answers.input_ids)
        answers[answers == tokenizer.pad_token_id] = -100
        
        answers = answers.to(device)
        global_weights = inputs.pop("global_weights").to(device)
        local_weights = inputs.pop("local_weights").to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        visuals_global = inputs.pop("visuals_global").to(device)
        visuals_local = inputs.pop("visuals_local").to(device)
        
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        visuals_global=visuals_global,
                        visuals_local=visuals_local,
                        global_weights=global_weights,
                        local_weights=local_weights,
                        labels=answers
                        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        epoch_loss.append(loss.item())
        if writer:
            writer.add_scalar(
                "batch loss / train", loss.item(), epoch * len(train_dataloader) + i
            )
            
    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    if writer:
        writer.add_scalar("loss / train", avg_loss, epoch)

    return metrics_best