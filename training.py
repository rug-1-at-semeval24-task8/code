import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def train_loop(
    dataloader, model, optimizer, scheduler, device, local_device, skip_visual=False
):
    print("Training...")
    model.train()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    losses = []
    preds = []
    true_Y = []
    for XY in dataloader:
        raw_test = XY["Text"]
        XY = [XY["Tensor"].to(device), XY["Label"].to(device)]
        Xs = XY[:-1]
        Y = XY[-1]
        raw_pred = model(*Xs)
        preds.append(model.postprocessing(raw_pred))
        true_Y.append(Y.to(local_device).numpy())
        loss = model.compute_loss(raw_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().to(local_device).numpy())
        progress_bar.update(1)
    scheduler.step()
    print("Train loss: " + str(np.mean(losses)))
    preds = np.concatenate(preds)
    true_Y = np.concatenate(true_Y)
    f1 = f1_score(y_true=true_Y, y_pred=preds, average="macro", zero_division=1)
    return np.mean(losses), f1


def eval_loop(dataloader, model, device, local_device, skip_visual=False, test=False):
    if test:
        print("Generating predictions...")
    else:
        print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    preds = []
    probs = []
    true_Y = []
    losses = []
    with torch.no_grad():
        for XY in dataloader:
            raw_test = XY["Text"]
            XY = [XY["Tensor"].to(device), XY["Label"].to(device)]
            Xs = XY[:-1]
            expected = XY[-1].detach().clone()
            Y = XY[-1].to(local_device)
            output = model(*Xs, raw_test)
            pred = model.postprocessing(output, argmax=True)
            preds.append(pred)
            prob = np.exp(model.postprocessing(output, argmax=False))
            probs.append(prob)
            Y = Y.numpy()
            true_Y.append(Y)
            eq = np.equal(Y, pred)
            size += len(eq)
            correct += sum(eq)
            loss = model.compute_loss(output, expected)
            losses.append(loss.detach().to(local_device).numpy())
            progress_bar.update(1)
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    true_Y = np.concatenate(true_Y)
    if not test:
        print("Accuracy: " + str(correct / size))
        f1_micro = f1_score(y_true=true_Y, y_pred=preds, average="micro", zero_division=1)
        f1_macro = f1_score(y_true=true_Y, y_pred=preds, average="macro", zero_division=1)
        print("F1-micro score: " + str(f1_micro))
        print("F1-macro score: " + str(f1_macro))
        return f1_micro, f1_macro, np.mean(losses)
    else:
        return preds, probs
