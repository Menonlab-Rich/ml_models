import logging
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from dataset import GenericDataset
from model import get_model
from config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, INPUT_LOADER, \
    TARGET_LOADER, TRANSFORMS, LOSS_FN, MULTI_GPU, NUM_CLASSES, \
    MODEL_PATH, CLASS_MAPPING, PREDICTIONS_PATH
import utils
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def train(model, loader, optimizer, scaler):
    logger.info('Training ResNet model')
    model.train()
    model.to(DEVICE)
    progress_bar = tqdm(loader, total=len(loader))
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = LOSS_FN(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return loss.item()

def setup():
    model = get_model(num_classes=NUM_CLASSES)
    logger.info('Setting up training data')
    dataset = GenericDataset(input_loader=INPUT_LOADER, target_loader=TARGET_LOADER, transform=TRANSFORMS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    train_set, val_set = dataset.split(0.8)
    utils.save_dataset(train_set, 'train_set')
    utils.save_dataset(val_set, 'val_set')
    
    return model, train_set, val_set, optimizer, scaler, scheduler

def main():
    model, train_set, val_set, optimizer, scaler, scheduler = setup()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    best_loss = float('inf') # Initialize to infinity
    for epoch in range(EPOCHS):
        logger.info(f'Epoch {epoch + 1}/{EPOCHS}')
        train(model, train_loader, optimizer, scaler)
        val_loss = utils.evaluate(model, val_loader, LOSS_FN, DEVICE)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            utils.save_model(model, MODEL_PATH)
            logger.info('Model saved')
            utils.plot_model_predictions(model, val_loader, CLASS_MAPPING, device=DEVICE, output_path=PREDICTIONS_PATH)
        
    
    