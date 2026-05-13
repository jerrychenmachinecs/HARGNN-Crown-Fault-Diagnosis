import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
from models import HARGNNClassifier
from loss import ENSCostSensitiveLoss
from dataset import HotRolledSteelDataset, collate_temporal_graphs
from torch.utils.data import DataLoader

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, prec_macro, rec_macro, f1_macro, f1_weighted, mcc, kappa

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels =[]
    for batch_graphs, labels in loader:
        batch_graphs = [g.to(device) for g in batch_graphs]
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch_graphs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc, prec, rec, f1_mac, f1_w, mcc, kappa = calculate_metrics(all_labels, all_preds)
    return total_loss / len(loader), acc, f1_mac

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels =[]
    with torch.no_grad():
        for batch_graphs, labels in loader:
            batch_graphs = [g.to(device) for g in batch_graphs]
            labels = labels.to(device)
            logits, _ = model(batch_graphs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc, prec, rec, f1_mac, f1_w, mcc, kappa = calculate_metrics(all_labels, all_preds)
    return total_loss / len(loader), acc, prec, rec, f1_mac, f1_w, mcc, kappa

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HotRolledSteelDataset('data/train.csv')
    val_dataset = HotRolledSteelDataset('data/val.csv')
    test_dataset = HotRolledSteelDataset('data/test.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_temporal_graphs)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_temporal_graphs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_temporal_graphs)
    
    samples_per_cls = torch.tensor([4520, 85710, 9770], dtype=torch.float32)
    criterion = ENSCostSensitiveLoss(samples_per_cls=samples_per_cls).to(device)
    
    model = HARGNNClassifier(node_input_dim=3, gnn_hidden_dim=128, gru_hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best_f1 = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(200):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1_mac, val_f1_w, val_mcc, val_kappa = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1_mac:.4f}")
        
        if val_f1_mac > best_f1:
            best_f1 = val_f1_mac
            torch.save(model.state_dict(), 'best_hargnn.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    model.load_state_dict(torch.load('best_hargnn.pth'))
    test_loss, t_acc, t_prec, t_rec, t_f1_mac, t_f1_w, t_mcc, t_kappa = evaluate(model, test_loader, criterion, device)
    print(f"Test Results -> Acc: {t_acc:.4f}, Macro-F1: {t_f1_mac:.4f}, Weighted-F1: {t_f1_w:.4f}, MCC: {t_mcc:.4f}, Kappa: {t_kappa:.4f}")

if __name__ == "__main__":
    main()
