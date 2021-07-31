import torch
import numpy as np

class MLPCritic():
  def __init__(self, params):
    self.ob_dim = params["ob_dim"]
    self.n_layers = params["n_layers"]
    self.hidden_layer_size = params["hidden_layer_size"]
    self.learning_rate = params["learning_rate"]
    self.device = torch.device('cpu')

    # MLP model
    layers = []
    in_size = self.ob_dim
    for _ in range(self.n_layers):
      layers.append(torch.nn.Linear(in_size, self.hidden_layer_size))
      layers.append(torch.nn.ReLU())
      in_size = self.hidden_layer_size
    layers.append(torch.nn.Linear(in_size, 1))
    self.value_function_nn = torch.nn.Sequential(*layers).to(self.device)
    
    # Adam Optimizer
    self.optimizer = torch.optim.Adam(self.value_function_nn.parameters(), self.learning_rate)
  
  def get_value(self, ob):
    ob_tensor = torch.tensor(ob, dtype=torch.float, device=self.device)
    value_tensor = self.value_function_nn(ob_tensor)
    return value_tensor.cpu().detach().numpy()

  def update(self, obs, q_values):
    x = np.concatenate(tuple(obs))
    x_tensor = torch.tensor(x, dtype=torch.float, device=self.device)
    # print(f"x_tensor: {x_tensor.shape}")
    y = np.concatenate(tuple(q_values))
    y_tensor = torch.tensor(y, dtype=torch.float, device=self.device)
    # print(f"y_tensor: {y_tensor.shape}")

    y_pred = self.value_function_nn(x_tensor)[:,0]
    # print(f"y_pred: {y_pred[:,0].shape}")

    loss = torch.nn.functional.mse_loss(y_tensor, y_pred)
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()




















