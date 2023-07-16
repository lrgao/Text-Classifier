import numpy as np
import torch
import json

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification as MyBert

from data import CLSDataLoader, CLSDataset
from tqdm import tqdm, trange


def run(args, logger):
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    if args.do_train:
        # Finetune
        train_dataset = CLSDataset(args, args.train_file, tokenizer, "train", attribute=args.attribute)
        dev_dataset = CLSDataset(args, args.predict_file, tokenizer, "val", attribute=args.attribute)
        test_dataset = CLSDataset(args, args.test_file, tokenizer, "val", attribute=args.attribute)
        train_dataloader = CLSDataLoader(args, train_dataset, "train")
        dev_dataloader = CLSDataLoader(args, dev_dataset, "val")
        test_dataloader = CLSDataLoader(args, test_dataset, "val")
    else:
        # Inference
        dev_dataset = CLSDataset(args, args.predict_file, tokenizer, "val", attribute=args.attribute)
        dev_dataloader = CLSDataLoader(args, dev_dataset, "val")

    if args.do_train:
        # Load model parameters
        model = MyBert.from_pretrained(args.model_path, config=args.model_config)

        print('model parameters: ', model.num_parameters())
        
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if not args.no_lr_decay:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=t_total)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=1000000)

        train(args, logger, model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, tokenizer)

    if args.do_predict:
        # Inference on the test set
        checkpoint = args.output_dir
        model = MyBert.from_pretrained(checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=True, infer_file=args.predict_file)
        logger.info("%s on %s data: %.2f" % (dev_dataloader.dataset.metric, dev_dataloader.dataset.data_type, ems*100))


def train(args, logger, model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, tokenizer):
    model.train()
    global_step = 0
    wait_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training = False

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting training!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            if global_step == 1:
                for tmp_id in range(len(batch)):
                    print(batch[tmp_id])

            output = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            
            loss = output.loss
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            # Gradient accumulation
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            # Print loss and evaluate on the valid and test set
            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = inference(model if args.n_gpu == 1 else model.module, dev_dataloader, tokenizer, args, logger, infer_file=args.predict_file)
                test_em = inference(model if args.n_gpu == 1 else model.module, test_dataloader, tokenizer, args, logger,infer_file=args.test_file)

                logger.info("Step %d Train loss %.2f Learning rate %.2e val %s %.2f%% test %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    scheduler.get_lr()[0],
                    'acc',
                    curr_em * 100,
                    'acc',
                    test_em * 100,
                    epoch))
                
                train_losses = []
                if best_accuracy < curr_em:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_dir)
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                                (dev_dataloader.dataset.metric, best_accuracy * 100.0, curr_em * 100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                    
                    # model_to_save = model.module if hasattr(model, 'module') else model
                    # model_to_save.save_pretrained(args.output_dir)
                    # logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                    #             (dev_dataloader.dataset.metric, best_accuracy * 100.0, curr_em * 100.0, epoch, global_step))
                model.train()
        if stop_training:
            break


def inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=False, infer_file='test.json'):
    # Inference on the test set
    is_rights = []
    pred_label_ids = []
    
    dev_iterator = tqdm(dev_dataloader, desc="Inference")
    for batch in dev_iterator:
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
            
        output = model(input_ids=batch[0],
                       attention_mask=batch[1],
                       labels=batch[2])

        pred_label_id = torch.argmax(output.logits, dim=-1)
        labels = batch[2].squeeze()
        is_right = (pred_label_id == labels)
        is_rights.append(is_right)
        pred_label_ids.append(pred_label_id)

    is_rights = torch.cat(is_rights, dim=0).int()
    pred_label_ids = torch.cat(pred_label_ids, dim=0).cpu().numpy().tolist()

    acc = torch.sum(is_rights)/is_rights.shape[0]
    acc = acc.cpu().numpy().item()
    
    if save_predictions:
        test_list = [data_ele for data_ele in dev_dataloader.dataset.data]
        for i, pred in enumerate(pred_label_ids):

            predlabel = dev_dataloader.dataset.id2labels[pred]
            attribute = args.attribute
            
            test_list[i]['pred_'+attribute] = predlabel
            if predlabel == test_list[i][attribute]:
                test_list[i]['pred_isright_'+attribute] = 1
            else:
                test_list[i]['pred_isright_'+attribute] = 0

        json.dump(test_list, open(infer_file+'.pred.json', 'w', encoding='utf-8'), indent=3, ensure_ascii=False)
        print('pred result saved in:', infer_file+'.pred.json')
    return acc
