from pytorch_lightning import Trainer
from dataset import ResnetDataModule, InputLoader, TargetLoader
from config import Config
from pytorch_lightning.loggers import NeptuneLogger
from tqdm import tqdm
from torch import sigmoid, tensor
import pandas as pd
from torch.nn import BCEWithLogitsLoss


def load_models(resnet_ckpt_path):
    from model import ResNet
    model = ResNet.load_from_checkpoint(resnet_ckpt_path, encoder=None, strict=False)
    return model




def main(config: Config):
    
    resnet_ckpt_path = r"D:\CZI_scope\code\ml_models\resnet\checkpoints\resnet-epoch=05-accuracy_val=0.98.ckpt"
    model = load_models(resnet_ckpt_path)
    model.eval()
    input_loader = InputLoader(config.test_dir) 
    target_loader = TargetLoader(config.test_dir, config.classes)
    transform = config.transform
    table = pd.DataFrame(columns=["filename", "prediction", "target", "loss"])
    for i, img in tqdm(enumerate(input_loader), total=len(input_loader), desc="Predicting"):
        img = transform.apply_val(img)
        img = img.unsqueeze(0)
        target = transform.apply_val(target_loader[i], input=False).unsqueeze(0)
        loss_fn = BCEWithLogitsLoss()
        output = model(img)
        loss = loss_fn(output, target)
        output = sigmoid(output)
        output = 1 if output.item() > 0.5 else 0
        filename = input_loader.get_ids(i)[0]
        table.loc[len(table.index)] = [filename, output, target.item(), loss.item()]
    
    # average accuracy
    accuracy = (table["prediction"] == table["target"]).mean()
    # average loss
    loss = table["loss"].mean()
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    
    

if __name__ == '__main__':
    config = Config(r'D:\CZI_scope\code\ml_models\resnet\config.yml')
    main(config)