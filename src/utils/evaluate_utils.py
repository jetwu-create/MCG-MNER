import torch
from tqdm import tqdm
from utils.PredictionSpan import PredictionSpanFormatter
from utils.metrics import calculate_metrics
import pandas as pd
import json

prediction_span_formatter = PredictionSpanFormatter()

def evaluate(
    model,
    tokenizer,
    dataloader,
    writer,
    device,
    epoch,
    generation_kwargs,
    options,
):
    model.eval()
    
    epoch_loss, spans_true,spans_pred = [], [], []
    with torch.no_grad():
        for i, inputs in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Evaluating",
        ):
            instances = inputs.pop("instances")
            contexts = [instance.context for instance in instances]
            spans_true_batch = [instance.entity_spans for instance in instances]
            spans_true.extend(spans_true_batch)
            
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
            
            outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        visuals_global=visuals_global,
                        visuals_local=visuals_local,
                        global_weights=global_weights,
                        local_weights=local_weights,
                        labels=answers
                        )
            
            loss = outputs.loss
            prediction_texts = model.generate(
                                            input_ids=input_ids, 
                                            attention_mask=attention_mask,
                                            visuals_global=visuals_global,
                                            visuals_local=visuals_local,
                                            global_weights=global_weights,
                                            local_weights=local_weights,
                                            **generation_kwargs,
                                            )
            
            prediction_texts = tokenizer.batch_decode(
                prediction_texts,
                skip_special_tokens=True,
            )
            
            if writer:
                writer.add_text('sample_prediction', prediction_texts[0])
                
            spans_pred_batch = [
                prediction_span_formatter.format_answer_spans(
                    context, prediction, options
                )
                for context, prediction in zip(contexts, prediction_texts)
            ]
            
            spans_pred.extend(spans_pred_batch)
            
            batch_metrics = calculate_metrics(
                spans_pred_batch, spans_true_batch, options=options
            )
            
            if writer:
                for metric_class, metric_dict in batch_metrics.items():
                    writer.add_scalars(
                        metric_class, metric_dict, epoch * len(dataloader) + i
                    )

            epoch_loss.append(loss.item())
            
            if writer:
                writer.add_scalar(
                    "batch loss / evaluation", loss.item(), epoch * len(dataloader) + i
                )
                
        epoch_metrics = calculate_metrics(spans_pred, spans_true, options=options)
        show_classification_report(epoch_metrics)
            
        return epoch_metrics
    

def update_best_checkpoint(
    metrics_new,
    metrics_best,
    metric_name,
    metric_avg,
    model,
    tokenizer,
):
    metric_current_value = metrics_new[metric_avg][metric_name]
    metric_best_value = 0.0
    
    if len(metrics_best) > 0:
        metric_best_value = metrics_best[metric_avg][metric_name]
    if metric_current_value > metric_best_value:
        print(
            f"Got Better results for {metric_name}. \n"
            f"{metric_current_value} > {metric_best_value}. Updating the best checkpoint"
        )
        metrics_best = metrics_new

    return metrics_best
        
        
def show_classification_report(metrics):
    df = pd.DataFrame.from_dict(metrics)
    print(df.transpose())