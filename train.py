
import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    batch_size=16,
    lr=0.001,
    weight_decay=0.0,
    grad_clip=1.0,
    model_path="best_model.pt",
    model_name=None,
    device=None,
    step_lr_step=10,
    step_lr_gamma=0.5,
    task="regression",
    pos_weight=None,
    early_stopping_patience=5,
    sample_weights=None,
    debug=False,
    scheduler_type="plateau",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if task == "classification":
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        else:
            pos_weight_tensor = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
    else:
        criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_lr_step, gamma=step_lr_gamma
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    best_val_loss = float("inf")

    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i in tqdm(
            range(0, len(X_train), batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
            unit="batch",
        ):
            X_batch = X_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if sample_weights is not None:
                weight_batch = sample_weights[i:i+batch_size].to(device)
                loss = (loss * weight_batch).mean()
            else:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            if debug and epoch == 0 and i == 0:
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                print(
                    f"Debug: outputs mean/std {outputs.mean().item():.4f}/"
                    f"{outputs.std().item():.4f}, y mean/std "
                    f"{y_batch.mean().item():.4f}/{y_batch.std().item():.4f}, "
                    f"grad_norm {grad_norm:.4f}"
                )
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_train_loss = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            X_val_device = X_val.to(device)
            y_val_device = y_val.to(device)
            val_outputs = model(X_val_device)
            val_loss = criterion(val_outputs, y_val_device).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint = {
                "model_name": getattr(model_name, "value", model_name) or model.__class__.__name__,
                "model_structure": str(model),
                "state_dict": model.state_dict(),
            }
            torch.save(checkpoint, model_path)
        else:
            epochs_no_improve += 1

        if scheduler_type == "step":
            scheduler.step()
        else:
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Best Val: {best_val_loss:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
    except Exception as exc:
        print(f"Warning: failed to load checkpoint '{model_path}': {exc}")
    return model.to(device)
