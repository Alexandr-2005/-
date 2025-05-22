import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

class MLP(nn.Module):

    def __init__(self, layer_sizes, dropout=0.0, batch_norm=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i+1]
            fc = nn.Linear(in_size, out_size)
            nn.init.kaiming_normal_(fc.weight, nonlinearity='relu')
            nn.init.zeros_(fc.bias)
            layers.append(fc)
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU(inplace=True))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_size))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Trainer:

    def __init__(self, model, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def fit(self,
            train_dataset,
            val_dataset=None,
            batch_size=64,
            epochs=20,
            lr=1e-3,
            optimizer_name='adam',
            weight_decay=0.0,
            lr_decay=1.0,
            early_stopping=False,
            patience=5):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None

        criterion = nn.CrossEntropyLoss()
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [],   'val_acc': []
        }
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.size(0), -1).to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_corrects = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.view(inputs.size(0), -1).to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item() * labels.size(0)
                        preds = outputs.argmax(dim=1)
                        val_corrects += (preds == labels).sum().item()
                        val_total += labels.size(0)

                val_epoch_loss = val_loss / val_total
                val_epoch_acc = val_corrects / val_total
                history['val_loss'].append(val_epoch_loss)
                history['val_acc'].append(val_epoch_acc)

                print(f"Epoch {epoch}/{epochs} | "
                      f"Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.2%} | "
                      f"Val   loss: {val_epoch_loss:.4f}, acc: {val_epoch_acc:.2%}")

                if early_stopping:
                    if val_epoch_acc > best_acc:
                        best_acc = val_epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            print("Early stopping triggered, restoring best model weights.")
                            self.model.load_state_dict(best_model_wts)
                            break
            else:
                print(f"Epoch {epoch}/{epochs} | Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.2%}")

            scheduler.step()

        if val_loader is not None:
            self.model.load_state_dict(best_model_wts)

        return history

    def predict(self, dataset, batch_size=64):
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.view(inputs.size(0), -1).to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1).cpu().tolist()
                predictions.extend(preds)
        return predictions


def save_model(model, path):

    torch.save(model.state_dict(), path)


def load_model(
    path,
    layer_sizes=[28*28, 512, 256, 128, 10],
    dropout=0.5,
    batch_norm=True,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(layer_sizes, dropout=dropout, batch_norm=batch_norm).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
