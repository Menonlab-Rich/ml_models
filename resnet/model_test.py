import unittest
import torch
from torch.utils.data import DataLoader
from model import BCEResnet
from base.dataset import MockDataLoader, GenericDataset
import dataset as ds

class MockResnetDataset(GenericDataset):
    def __init__(self, input_loader, target_loader, transform=None, **kwargs):
        super().__init__(input_loader, target_loader, transform)

    def __getitem__(self, idx):
        input_data = self.input_loader[idx]
        target_data = torch.randint(0, 2, (1,)).float()
        return input_data, target_data, idx

# Monkey patch the ResnetDataset class with the mock class
ds.ResnetDataset = MockResnetDataset

class TestResnetDataModule(unittest.TestCase):
    def setUp(self):
        # Mock data
        self.input_data = torch.randn(100, 3, 64, 64)
        self.target_data = torch.randint(0, 2, (100, 1)).float()

        # Mock loaders
        self.input_loader = MockDataLoader(self.input_data)
        self.target_loader = MockDataLoader(self.target_data)

        # DataModule instance
        self.datamodule = ds.ResnetDataModule(
            input_loader=self.input_loader, 
            target_loader=self.target_loader, 
            batch_size=10, 
            n_workers=2
        )

    def test_setup(self):
        self.datamodule.setup('fit')
        self.assertIsInstance(self.datamodule.train_set, MockResnetDataset)
        self.assertIsInstance(self.datamodule.val_set, MockResnetDataset)

    def test_dataloader(self):
        self.datamodule.setup('fit')
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()

        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        self.assertEqual(len(train_batch), 3)  # input, target, index
        self.assertEqual(len(val_batch), 3)  # input, target, index

    def test_state_dict(self):
        state = self.datamodule.state_dict()
        self.datamodule.batch_size = 20
        self.datamodule.load_state_dict(state)
        self.assertEqual(self.datamodule.batch_size, 10)

class TestResNet(unittest.TestCase):
    def setUp(self):
        self.model = BCEResnet(n_classes=1, n_channels=3, lr=1e-3)
        self.batch = (torch.randn(10, 3, 224, 224), torch.randint(0, 2, (10, 1)).float(), None)

    def test_forward(self):
        x = torch.randn(2, 3, 224, 224)
        y = self.model(x)
        self.assertEqual(y.shape, (2, 1))

    def test_training_step(self):
        loss = self.model.training_step(self.batch, 0)
        self.assertTrue(loss.requires_grad)

    def test_validation_step(self):
        with torch.no_grad():
            loss = self.model.validation_step(self.batch, 0)
        self.assertFalse(loss.requires_grad)

    def test_test_step_accuracy(self):
        # Reset metrics
        self.model.validation_accuracy.reset()
        self.model.test_accuracy.reset()

        # Run validation step
        with torch.no_grad():
            self.model.validation_step(self.batch, 0)
            val_accuracy = self.model.validation_accuracy.compute()

        # Run test step
        with torch.no_grad():
            self.model.test_step(self.batch, 0)
            test_accuracy = self.model.test_accuracy.compute()

        # Check that the accuracies are the same
        self.assertAlmostEqual(val_accuracy.item(), test_accuracy.item(), places=5)

    def test_configure_optimizers(self):
        optim = self.model.configure_optimizers()
        self.assertIsInstance(optim, torch.optim.Adam)

if __name__ == '__main__':
    unittest.main()
