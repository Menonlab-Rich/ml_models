import config
import torch
from torchvision.utils import save_image

def save_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5 # remove normalization
        save_image(y_fake, f"{folder}/y_gen_{epoch}.png")
        save_image(x, f"{folder}/x_input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, f"{folder}/labelt_{epoch}.png")                
    
    gen.train()
    
def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    

def load_checkpoint(model, optimizer, filename, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # if we don't do this then lr might be different
    # because the optimizer might have a different state
    # than the one we saved.
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr