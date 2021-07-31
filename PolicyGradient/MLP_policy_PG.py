import torch
import numpy as np

class MLPPolicyPG():
  def __init__(self, params):
    self.ac_dim = params["ac_dim"]
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
    layers.append(torch.nn.Linear(in_size, self.ac_dim))
    self.logits_nn = torch.nn.Sequential(*layers).to(self.device)
    
    # Adam Optimizer
    self.optimizer = torch.optim.Adam(self.logits_nn.parameters(), self.learning_rate)

  def get_action(self, ob):
    ac = self.get_action_distribution(ob).sample()
    return ac.cpu().detach().numpy()

  def get_action_distribution(self, ob):
    ob_tensor = torch.tensor(ob, dtype=torch.float, device=self.device)
    ac_distribution = torch.distributions.Categorical(logits=self.logits_nn(ob_tensor))
    return ac_distribution

  def update(self, trajectories, advantages):
    loss = torch.tensor(0, dtype=torch.float, device=self.device)
    N = len(trajectories["rewards"]) # Number of Trajectories

    for i in range(N):
      obs = trajectories["obs"][i]

      acs = trajectories["acs"][i]
      acs_tensor = torch.tensor(acs, dtype=torch.float, device=self.device)

      acs_distribution = self.get_action_distribution(obs)
      log_probs = acs_distribution.log_prob(acs_tensor)

      #   "advantages": [np.array[a1, a2, a3, ...], np.array[a1, a2, a3, ...], ...]
      advantages_tensor = torch.tensor(advantages[i], dtype=torch.float, device=self.device)
      
      loss -= torch.dot(log_probs, advantages_tensor)

    loss /= N
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()










