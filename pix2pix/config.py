import yaml
import torch
from base.config import BaseConfigHandler, create_transform_function
from albumentations import ImageOnlyTransform


class CustomTransforms:
    '''
    A namespace for custom transforms
    '''

    def __init__(self):
        pass
    
    class A_ToTensorWithDType(ImageOnlyTransform):
        def __init__(self, always_apply=True, p=1.0, dtype=torch.float32):
            super(CustomTransforms.A_ToTensorWithDType, self).__init__(
                always_apply=always_apply, p=p)
            if type(dtype) == str:
                self.dtype = getattr(torch, dtype)
            self._targets_as_params = None

        def apply(self, target, **params):
            x = torch.tensor(target, dtype=self.dtype)
            x = torch.unsqueeze(x, 0) # Add a channel dimension
            return x

    class TV_ToTensorWithDType():
        def __init__(self, dtype=torch):
            self.dtype = getattr(torch, dtype)

        def __call__(self, target):
            return torch.tensor(target, dtype=self.dtype)


class Config(BaseConfigHandler):
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = self.parse_config(f)

    def parse_config(self, f):
        handle_special = {
            'optimizer': self.handle_optimizer,
            'scheduler': self.handle_scheduler,
            'transform': self.handle_transforms,
            'device': self.handle_device,
        }
        unparsed = yaml.load(f, Loader=yaml.FullLoader)
        config = {}
        for key, value in unparsed.items():
            if key in handle_special:
                config[key] = handle_special[key](value)
            else:
                config[key] = value

        return config

    def handle_device(self, value):
        if type(value) == str:
            return value

        default = value['default']

        # If the user wants to use cuda if available, check if cuda is available
        # and return cuda, else return default
        if value['cuda_if_available']:
            return 'cuda' if torch.cuda.is_available() else default

        return default

    def handle_optimizer(self, value):
        if value['kind'].lower() == 'adam':
            return lambda p: torch.optim.Adam(p, **value['params'])
        elif value['kind'].lower() == 'sgd':
            return lambda p: torch.optim.SGD(p, **value['params'])
        else:
            raise ValueError(f"Unknown optimizer {value['kind']}")

    def handle_scheduler(self, value):
        if value['kind'].lower() == 'step':
            return lambda o: torch.optim.lr_scheduler.StepLR(o, **value['params'])
        elif value['kind'].lower() == 'plateau':
            return lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o, **value['params'])
        else:
            raise ValueError(f"Unknown scheduler {value['kind']}")

    def handle_transforms(self, value):
        if 'lib' in value and value['lib'].lower() == 'torchvision':
            import torchvision.transforms as transforms
            lib = 'torchvision'
        elif 'lib' in value and value['lib'].lower() == 'albumentations':
            import albumentations as transforms
            lib = 'albumentations'
        elif 'lib' in value:
            raise ValueError(f"Unknown library {value['lib']}")
        else:
            lib = 'torchvision'
            import torchvision.transforms as transforms  # default to torchvision

        transform = {
            'train': {
                'input': None,
                'target': None,
            },
            'val': {
                'input': None,
                'target': None,
            }
        }

        def handle_transform(conf, training_pipeline, validation_pipeline):
            args = conf.get('args', {}) or {}
            if 'custom' in conf and conf['custom']:
                if lib == 'albumentations':
                    name = 'A_' + conf['xform']
                else:
                    name = 'TV_' + conf['xform']
                xform = getattr(CustomTransforms, name)(**args)
            else:
                xform = getattr(transforms, conf['xform'])(**args)
            if 'pipeline' in conf and conf['pipeline'] == 'both':
                training_pipeline.append(xform)
                validation_pipeline.append(xform)
            elif 'pipeline' in conf and conf['pipeline'] == 'validation':
                validation_pipeline.append(xform)
            else:
                training_pipeline.append(xform)

        train_input_pipeline = []
        train_target_pipeline = []
        validation_input_pipeline = []
        validation_target_pipeline = []

        if 'input' in value:
            for val in value['input']:
                handle_transform(val, train_input_pipeline,
                                 validation_input_pipeline)
        if 'target' in value:
            for val in value['target']:
                handle_transform(val, train_target_pipeline,
                                 validation_target_pipeline)

        train_input_pipeline = transforms.Compose(train_input_pipeline)
        train_target_pipeline = transforms.Compose(train_target_pipeline)
        validation_input_pipeline = transforms.Compose(
            validation_input_pipeline)
        validation_target_pipeline = transforms.Compose(
            validation_target_pipeline)
        train_input_transform = create_transform_function(
            value['targets']['input'],
            'input_pipeline', train_input_pipeline)
        train_target_transform = create_transform_function(
            value['targets']['target'],
            'target_pipeline', train_target_pipeline)
        validation_input_transform = create_transform_function(
            value['targets']['input'], 'input_pipeline', validation_input_pipeline)
        validation_target_transform = create_transform_function(
            value['targets']['target'],
            'target_pipeline', validation_target_pipeline)

        transform['train']['input'] = train_input_transform
        transform['train']['target'] = train_target_transform
        transform['val']['input'] = validation_input_transform
        transform['val']['target'] = validation_target_transform

        return transform

    def load(self, path: str):
        return super().load(path)

    def save(self, path: str):
        return super().save(path)


pixel_wise_loss = torch.nn.MSELoss()
adversarial_loss = torch.nn.BCEWithLogitsLoss()


class GeneratorLoss(torch.nn.Module):
    def __init__(self, lambda_=0.001):
        super(GeneratorLoss, self).__init__()
        self.lambda_adv = lambda_

    def forward(self, fake, generated, real):
        truthy = torch.ones_like(fake)
        adv_loss = adversarial_loss(fake, truthy)
        pixel_loss = pixel_wise_loss(generated, real)
        return adv_loss * self.lambda_adv + pixel_loss


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, output, is_real):
        return adversarial_loss(
            output, torch.ones_like(output)
            if is_real else torch.zeros_like(output))
