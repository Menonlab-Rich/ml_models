from dataset import ResnetDataModule, InputLoader, TargetLoader



class ResnetDataset:
    def __init__(self, inputs, targets, transforms):
        self.inputs = inputs
        self.targets = targets
        self.transforms = transforms

# Your ResnetDataModule class here

def test_resnet_data_module():
    input_data = ['input1', 'input2', 'input3', 'input4', 'input5']
    target_data = ['target1', 'target2', 'target3', 'target4', 'target5']
    
    input_loader = InputLoader(directory='.', files=input_data)
    target_loader = TargetLoader(directory='.', files=target_data, class_labels=['tar'])
    
    datamodule = ResnetDataModule(
        input_loader=input_loader, 
        target_loader=target_loader, 
        batch_size=64, 
        n_workers=4
    )

    # Save the state
    state_dict = datamodule.state_dict()

    # Modify the attributes
    datamodule.batch_size = 128
    datamodule.n_workers = 2

    # Load the state
    datamodule.load_state_dict(state_dict)

    # Verify the attributes are restored correctly
    assert datamodule.batch_size == 64, f"Expected batch_size to be 64 but got {datamodule.batch_size}"
    assert datamodule.n_workers == 4, f"Expected n_workers to be 4 but got {datamodule.n_workers}"
    assert datamodule.input_loader.files == input_data, "Input loader data mismatch"
    assert datamodule.target_loader.files == target_data, "Target loader data mismatch"
    
    print("All tests passed.")

# Run the test
if __name__ == '__main__':
    test_resnet_data_module()