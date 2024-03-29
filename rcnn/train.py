from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS, SAVE_ANNOTATED_IMAGES
)
from model import create_model
from utils import Averager, SaveBestModel, save_model, save_loss_plot, save_annotated_examples
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
import torch
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import time
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s - Line: %(lineno)d',
                    datefmt='%Y-%m-%d %H:%M:%S')

plt.style.use('ggplot')

# function for running training iterations
def train(train_data_loader, model, optimizer, accumulation_steps=2):
    logging.info('Training')
    global train_itr
    global train_loss_list

    model.train()
    scaler = GradScaler()  # Initialize the gradient scaler for AMP

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    accumulated_loss = 0.0  # To track loss over accumulation steps

    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # Scales the loss for automatic mixed precision
        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Scale the loss to prevent underflow
        scaler.scale(losses).backward()

        # Accumulate loss for reporting
        accumulated_loss += losses.item()

        # Only step and zero the gradients every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update the scaler
            optimizer.zero_grad()  # Reset gradients
            
            # Log the accumulated loss averaged over accumulation steps and reset
            loss_value = accumulated_loss / accumulation_steps
            train_loss_list.append(loss_value)
            train_loss_hist.send(loss_value)
            accumulated_loss = 0.0  # Reset accumulated loss
            
            # Update progress bar description with the latest loss
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
            train_itr += 1

    # Handle any remaining accumulated gradients if the dataset size
    # isn't perfectly divisible by accumulation_steps * batch_size
    if len(train_data_loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # No need to log loss here as it's done in the loop

    return train_loss_list

# function for running validation iterations
def validate(valid_data_loader, model):
    logging.info('Validating')
    global val_itr
    global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

if __name__ == '__main__':
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    logging.info(f"Number of training samples: {len(train_dataset)}")
    logging.info(f"Number of validation samples: {len(valid_dataset)}\n")
    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_tranformed_image
        show_tranformed_image(train_loader)
        
        save_annotated_examples(model, valid_loader)
    # initialize SaveBestModel class
    save_best_model = SaveBestModel()
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        logging.info(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        logging.info(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        logging.info(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        logging.info(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # save the best model till now if we have the least loss in the...
        # ... current epoch
        save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )
        # save the current epoch model
        save_model(epoch, model, optimizer)
        # save loss plot
        save_loss_plot(OUT_DIR, train_loss, val_loss)
        if SAVE_ANNOTATED_IMAGES:
            save_annotated_examples(model, valid_loader, epoch)
