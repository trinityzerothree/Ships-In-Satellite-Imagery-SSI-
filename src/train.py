import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from dataloader.dataloader import load_shipsnet
from src.augmentations import train_transform
from src.model import BasicCNN

# This file trains the model and evaluates on the test set. It also prints out the confusion matrix and other metrics.
def eval_loop(model, loader, device):
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().tolist())
            y_all.extend(y.cpu().tolist())

    acc = accuracy_score(y_all, preds_all)
    f1  = f1_score(y_all, preds_all, average="binary")
    prec = precision_score(y_all, preds_all, pos_label=1)
    rec  = recall_score(y_all, preds_all, pos_label=1)
    cm  = confusion_matrix(y_all, preds_all)
    return acc, f1, prec, rec, cm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, test_loader = load_shipsnet(transform=train_transform())

    model = BasicCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_f1 = 0.0
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_acc, val_f1, val_prec, val_rec, val_cm  = eval_loop(model, val_loader, device)
        print(f"Epoch {epoch:02d} loss={total_loss/len(train_loader):.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_prec={val_prec:.4f} val_rec={val_rec:.4f}")
        print("Val confusion matrix [[TN FP],[FN TP]]:\n", val_cm)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_cnn.pt")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("best_cnn.pt", map_location=device))
    test_acc, test_f1, test_prec, test_rec, test_cm = eval_loop(model, test_loader, device)
    print("\nTEST RESULTS")
    print(f"test_acc={test_acc:.4f} test_f1={test_f1:.4f} test_prec={test_prec:.4f} test_rec={test_rec:.4f}")
    print("Test confusion matrix [[TN FP],[FN TP]]:\n", test_cm)

if __name__ == "__main__":
    main()