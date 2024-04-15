import utils
import config
import torch
from model import UNet


def main():
    model = UNet(in_channels=config.CHANNELS_INPUT,
                 out_channels=config.CHANNELS_OUTPUT).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Load the model
    utils.load_checkpoint(model, optimizer, config.CHECKPOINT)
    # plot the model
    utils.make_prediction(
        model,
        r"D:\CZI_scope\code\ml_models\unet\temp\tmpayw9b0vq\605-image_2024-04-03T20-38-24.745_0_page_868.tif",
        r"D:\CZI_scope\code\ml_models\unet\temp\tmpayw9b0vq\605-image_2024-04-03T20-38-24.745_0_page_868.tif.npz",
        r"D:\CZI_scope\code\ml_models\unet\results",      
        channels=(config.CHANNELS_INPUT, config.CHANNELS_OUTPUT),
        to_float=config.DATASET_TO_FLOAT,
        transforms=(config.transform_input, config.transform_target, config.transform_both))

if __name__ == "__main__":
    main()
