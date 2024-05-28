from dataset import ResnetDataModule, InputLoader, TargetLoader


def load_models(resnet_ckpt_path, encoder_ckpt_path):
    from model import ResNet
    from encoder.model import LitAutoencoder
    encoder = LitAutoencoder.load_from_checkpoint(encoder_ckpt_path, strict=False)
    encoder.eval() # Freeze the encoder (should be done already in the model but just in case)
    model = ResNet.load_from_checkpoint(resnet_ckpt_path, encoder=encoder, strict=False)
    return model



def main():
    resnet_ckpt_path = r"D:\CZI_scope\code\ml_models\resnet\best_model.ckpt"
    encoder_ckpt_path = r"D:\CZI_scope\code\ml_models\encoder\best_model.ckpt"
    model = load_models(resnet_ckpt_path, encoder_ckpt_path)
    input_loader = 

if __name__ == '__main__':
    main()