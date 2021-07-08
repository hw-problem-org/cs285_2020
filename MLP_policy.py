import torch

class MLPPolicy():
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

  def update(self, trajectories):
    raise NotImplementedError

class MLPPolicyREINFORCE(MLPPolicy):
  def __init__(self, params):
    super().__init__(params)

  def update(self, trajectories):
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

class MLPPolicyPG(MLPPolicy):
  def __init__(self, params):
    super().__init__(params)

  def update(self, trajectories):
    pass











