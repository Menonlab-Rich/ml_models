import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class _MockResource:
    '''
    A mock resource to be used when not using mixed precision training.
    It is used to allow the with statement to be used without any side effects.
    '''
    def __init__(self) -> None:
        pass
    
    # Allow the with statement to be used
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        pass # Nothing to do here
    
class _FlexibleScaler():
    class _MockTensor:
        '''
        Class to mock a tensor so that you can call backward on it.
        '''
        def __init__(self, loss) -> None:
            self.loss = loss
        def backward(self, *args, **kwargs):
            self.loss.backward()
    def __init__(self, device) -> None:
        self.scaler = GradScaler() if device == "cuda" else None
    
    def scale(self, loss):
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return self._MockTensor(loss) # Return a mock tensor so you can call backward on it
    
    def step(self, opt):
        if self.scaler is not None:
            self.scaler.step(opt) 
        else:
            opt.step()
    
    def update(self):
        if self.scaler is not None:
            self.scaler.update()

NUM_STEPS = 20e3 # The paper specifies training as global steps, not epochs
BATCH_SIZE = 128 # The paper specifies batch size of 128
LEARNING_RATE = 1e-3 # The paper specifies learning rate of 1e-3
BETAS = (0.5, 0.999) # The paper specifies betas of 0.5 and 0.999
NUM_WORKERS = 8 # Change this if not running on CHPC
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ORDINAL_REGRESSION_LOSS = nn.MSELoss() # The paper specifies the loss as Mean Squared Error (L2 loss)
AUTOCAST = autocast if DEVICE == "cuda" else _MockResource() # This will manage the autocasting transparently
GRAD_SCALER = _FlexibleScaler(DEVICE) # This will manage the gradient scaling transparently
