@torch.no_grad()
def check_lora(model):
    train_dataset, eval_dataset = PETDataset.from_folder('/root/inhousepet/train_data', tokenizer, max_len=1024, split=[0.9, 0.1])


    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=train_dataset.collate, batch_size=1
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=eval_dataset.collate, batch_size=8
    )

    # model.eval()
    losses = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        for key, value in batch.items():
            batch[key] = batch[key].cuda()

        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss.item()
        losses.append(loss)

        # print(loss)
        # break

    return losses
