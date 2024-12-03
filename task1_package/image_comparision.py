import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ImageSequencePredictor


def compare_predicted_and_target_images(model_path, data_loader, index):
    """
    Display comparison between predicted images and target images.

    Parameters:
    model (torch.nn.Module): The trained model path.
    data_loader (torch.utils.data.DataLoader): DataLoader with your dataset.
    index (int): The index of batch.
    """

    # Create the model
    input_dim = 64
    hidden_dim = [128, 64]
    kernel_size = (3, 3)
    num_layers = 2
    output_size = (360, 360)

    # Check for GPU
    device = 'cuda' if torch.cuda.device_count() > 0 and torch.cuda.is_available() else 'cpu' # noqa

    model_test = ImageSequencePredictor(input_dim, hidden_dim,
                                        kernel_size,
                                        num_layers,
                                        output_size)

    model_test.load_state_dict(torch.load(model_path, map_location=device))

    model_test = model_test.to(device)
    model_test.eval()

    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)

    # move inputs to the computation device
    inputs = inputs.to(device)

    # forward pass and compute the predictions
    with torch.no_grad():
        outputs = model_test(inputs)

    predicted_image = outputs[index].cpu().numpy()
    target_image = targets[index].cpu().numpy()
    input_image = np.squeeze(predicted_image)
    target_image = np.squeeze(target_image)

    fig, axes = plt.subplots(1, 6, figsize=(18, 6))

    # Iterate over the 3 generated images
    for i in range(3):
        axes[i].imshow(input_image[i], cmap='gray')
        axes[i].set_title(f'Generated Image {i+1}')
        axes[i].axis('off')

    # Iterate over the 3 target images
    for i in range(3):
        axes[i+3].imshow(target_image[i], cmap='gray')
        axes[i+3].set_title(f'Target Image {i+1}')
        axes[i+3].axis('off')

    plt.tight_layout()
    plt.show()
