import torch

class MLPPolicy():
  def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate):
    self.ac_dim = ac_dim
    self.ob_dim = ob_dim
    self.n_layers = n_layers
    self.size = size
    self.learning_rate = learning_rate
    self.device = torch.device('cpu')

    # MLP model
    layers = []
    in_size = ob_dim
    for _ in range(n_layers):
      layers.append(torch.nn.Linear(in_size, size))
      layers.append(torch.nn.ReLU())
      in_size = size
    layers.append(torch.nn.Linear(in_size, ac_dim))
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

  def update(self, trajectories):
    # trajectories: {
    #   "obs": [array[o1, o2, o3, ...], array[o1, o2, o3, ...], ...]
    #   "acs": [array[a1, a2, a3, ...], array[a1, a2, a3, ...], ...]
    #   "rewards": [array[r1, r2, r3, ...], array[r1, r2, r3, ...], ...]
    # }

    # Forward Pass
    loss = torch.tensor(0, dtype=torch.float, device=self.device)
    for i in range(len(trajectories["obs"])):
      obs = trajectories["obs"][i]
      acs = trajectories["acs"][i]
      acs_tensor = torch.tensor(acs, dtype=torch.float, device=self.device)
      rewards = trajectories["rewards"][i]
      rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)

      acs_distribution = self.get_action_distribution(obs)
      log_probs = acs_distribution.log_prob(acs_tensor)
      loss -= log_probs.sum() * rewards_tensor.sum()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()











