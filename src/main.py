import argparse
import numpy as np
import random
import torch
from pathlib import Path
from utils.utils import loads_json, load_json
import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import T5MNERDataset
from transformers import T5Tokenizer
from model.model import Mymodel
from utils.collator import Collator
from torch.utils.data import DataLoader
from utils.train_utils import train
from utils.evaluate_utils import evaluate
import logging as logging


seed = 666
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_n_batches", default=200, type=int)
    parser.add_argument("--pred_n_batches", default=200, type=int)
    parser.add_argument('--dataset', default='twitter2015', type=str)
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--lr', default=6e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--metric_name', default='f1-score', type=str)
    parser.add_argument('--metric_avg', default='micro_avg', type=str)
    parser.add_argument('--training', default=True, type=bool)
    parser.add_argument('--log', default=True, type=bool)
    parser.add_argument('--tokenizer_length', default=512, type=int)
    parser.add_argument('--padding', default=True, type=bool)
    parser.add_argument('--return_tensors', default='pt', type=str)
    parser.add_argument('--num_beams', default=2, type=int)
    parser.add_argument('--generation_length', default=128, type=int)
    args = parser.parse_args()
    
    set_global_seed(seed)          #seed setting
    
    v_g_w_path = Path(f'./data/weights/{args.dataset}_g_w.pth')
    v_l_w_path = Path(f'./data/weights/{args.dataset}_l_w.pth')
    
    v_g_w = torch.load(v_g_w_path)
    v_l_w = torch.load(v_l_w_path)
    
    v_g_img_path = Path(f'./data/{args.dataset}/global/')
    v_l_img_path = Path(f'./data/{args.dataset}/local/')
    
    json_path = f'./configs'
    json_path = Path(json_path)
    text_train_json = json_path / 'train.json'
    text_dev_json = json_path / 'valid.json'
    test_test_json = json_path / 'test.json'
    text_data_train = loads_json(text_train_json)
    text_data_valid = loads_json(text_dev_json)
    text_data_test = loads_json(test_test_json)
    
    tokenizer_kwargs = {
        'max_length': args.tokenizer_length, \
        'padding': args.padding, \
        'return_tensors': args.return_tensors
    }
    generation_kwargs = {
        'num_beams': args.num_beams, \
        'max_length': args.generation_length
    }
    
    writer = None
    if args.log:
        log_dir = Path(f'log') / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)
    
    labels_path = f'./configs/data_label.json'
    labels_path = Path(labels_path)
    label_set = load_json(labels_path)
    labels_ = label_set[args.dataset]
    
    instructions_path  = f'./configs/instructions_set.json'
    instructions_path = Path(instructions_path)
    instructions_ = load_json(instructions_path)
    
    train_dataset = T5MNERDataset(
                            data=text_data_train,
                            instructions=instructions_['train'],
                            options=labels_,
    )
    valid_dataset = T5MNERDataset(
                            data=text_data_valid,
                            instructions=instructions_['test'],
                            options=labels_,
    )
    test_dataset_ = T5MNERDataset(
                            data=text_data_test,
                            instructions=instructions_['test'],
                            options=labels_,
    )
    device = torch.device(f'cuda:{args.cuda}')
    
    model_path = f'./configs/t5_base'
    model_path = Path(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = Mymodel.from_pretrained(args, model_path, device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )
    
    collator = Collator(
        tokenizer=tokenizer,
        visual_global_path=v_g_img_path,
        visual_local_path=v_l_img_path,
        visual_global_weight=v_g_w,
        visual_local_weight=v_l_w,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collator,
        num_workers=8,
        pin_memory=True,
    )
    
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collator,
        num_workers=8,
        pin_memory=True,
    )
    
    test_dataloader_ = DataLoader(
        dataset=test_dataset_,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collator,
        num_workers=8,
        pin_memory=True,
    )
    
    if args.training:
        
        train(
            epochs=args.epochs,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            test_dataloader=valid_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device,
            eval_every_n_batches=args.eval_n_batches,
            pred_every_n_batches=args.pred_n_batches,
            generation_kwargs=generation_kwargs,
            options=labels_,
            metric_name_to_choose_best=args.metric_name,
            metric_avg_to_choose_best=args.metric_avg,
        )
        
    evaluate_metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=test_dataloader_,
                    writer=None,
                    device=device,
                    epoch=0,
                    generation_kwargs=generation_kwargs,
                    options=labels_,
                )
    
    result = evaluate_metrics['micro_avg']['f1-score']
    logging.basicConfig(filename=f'micro_f1_v1.log', level=logging.INFO)
    logging.info('micro-f1: {}'.format(result))
    
    
if __name__ == "__main__":
    main()