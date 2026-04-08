import torch
from eval_model import evaluate_model
from prepare_data import test_dataloader,test_data 
from visualization import show_predictions 
from Build_model import cnn_model
device = "cuda" if torch.cuda.is_available() else "cpu"

model=cnn_model(input_shape=3, hidden_units=32, output_shape=len(test_dataloader.dataset.classes)).to(device)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
acc, precision, recall, f1 = evaluate_model(model, test_dataloader, device)  

print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")   
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")



show_predictions(model, test_data, test_dataloader.dataset.classes, device)

