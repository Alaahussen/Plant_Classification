from typing import Dict, List
import matplotlib.pyplot as plt
import random
import math
import torch

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    plt.savefig("loss_curves.png")
    plt.show()



def show_predictions(model, dataset, class_names, device, n=10):
    """
    Show predictions for `n` random images from `dataset`.
    Displays 2 images per row with larger figure.
    """
    model.eval()

    indices = random.sample(range(len(dataset)), n)

    # Compute number of rows (2 images per row)
    n_rows = math.ceil(n / 2)
    plt.figure(figsize=(18, 6 * n_rows))  # bigger figure for long class names

    for i, idx in enumerate(indices):
        img, label = dataset[idx]

        input_img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)
            pred = torch.argmax(output, dim=1).item()

        # Convert tensor to HWC for plotting
        img_np = img.permute(1, 2, 0).cpu().numpy()

        plt.subplot(n_rows, 2, i+1)
        plt.imshow(img_np)
        plt.title(f"True: {class_names[label]}\nPred: {class_names[pred]}", fontsize=14)
        plt.axis("off")
    plt.savefig("res.png") 

    plt.tight_layout()
    plt.show()