from tqdm import tqdm
from config import num_epochs, loss_fn, lr, batch_size, device, \
    input_loader, target_loader, transforms, width, height, nclasses, scheduler 
from utils import split_loaders, plot_results, plot_loss
from ml_models.cnn.dataset import GenericDataset
from torch import optim
from torch.utils.data import DataLoader
from ml_models.cnn.model import CNN
from torch.cuda.amp import GradScaler, autocast
import logging

def train(train_loader, model, optimizer, val_loader=None, scheduler=None, scaler=None):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = []

        for inputs, targets in train_loader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Reset gradients for this iteration

            if device == 'cuda' and scaler is not None:
                with autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                scaler.scale(loss).backward()  # Scale loss and perform backward pass
                scaler.step(optimizer)  # Update model parameters
                scaler.update()  # Update the scale for next iteration
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss.append(loss.item())
            train_loader_tqdm.set_postfix(loss=np.sum(running_loss) / (len(inputs)), refresh=True)

        # Update the learning rate if a scheduler is provided
        if scheduler:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(running_loss / len(train_loader))
            else:
                scheduler.step()

        print(f'Epoch {epoch + 1} completed, Avg Loss: {running_loss / len(train_loader):.4f}')
        
        if val_loader:
            plot_results(model, val_loader)
    
    plot_loss(running_loss)

    print('Finished Training')

def main():
    logging.basicConfig(level=logging.INFO)
    dataset = GenericDataset(input_loader=input_loader, target_loader=target_loader, transform=transforms)
    train_loader, val_loader = split_loaders(dataset)
    model = CNN(width, height, nclasses).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler() if device == 'cuda' else None
    train(train_loader, model, optimizer, scheduler=scheduler(optimizer), scaler=scaler)

if __name__ == '__main__':
    main()
