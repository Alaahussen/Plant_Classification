import torch
import torch.nn as nn
from prepare_data import train_dataloader, test_dataloader, train_data
from Build_model import cnn_model
from Build_ANN import SimpleANN
from prepare_train import train
from visualization import plot_loss_curves
device = "cuda" if torch.cuda.is_available() else "cpu"


# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)
input_size = 256 * 256 * 3

# Set number of epochs
NUM_EPOCHS = 10

model_0 = cnn_model(input_shape=3,
                  hidden_units=32, 
                  output_shape=len(train_data.classes)).to(device)


model_ann=SimpleANN(input_size, len(train_data.classes)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)



model_0_results = train(model=model_0, 
                         train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,
                         optimizer=optimizer,
                         loss_fn=loss_fn, 
                         epochs=NUM_EPOCHS)


# model_ann_res=train(model=model_ann,
#                     train_dataloader=train_dataloader,
#                     test_dataloader=test_dataloader,
#                     optimizer=optimizer,
#                     loss_fn=loss_fn,
#                     epochs=NUM_EPOCHS)

torch.save(model_0.state_dict(), "cnn_model.pth")
plot_loss_curves(model_0_results)
