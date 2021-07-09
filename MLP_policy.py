import torch
import numpy as np

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

class MLPPolicyPG(MLPPolicy):
  def __init__(self, params):
    super().__init__(params)

    # param["variance_reduction"] can be:
    #   - "NONE"
    #   - "REWARD_TO_GO"
    #   - "BASELINE"
    #   - "BOTH"
    self.variance_reduction = params["variance_reduction"]

  def update(self, trajectories):
    loss = torch.tensor(0, dtype=torch.float, device=self.device)
    N = len(trajectories["obs"]) # Number of Trajectories

    # Baseline Calculation
    if self.variance_reduction == "BASELINE":
      baseline = 0
      for i in range(N):
        rewards = trajectories["rewards"][i]
        baseline += rewards.sum()
      baseline /= N
    elif self.variance_reduction == "BOTH":
      baseline = []
      ntraj_contribute = []
      for i in range(N):
        rewards = trajectories["rewards"][i]
        l = rewards.shape[0] # Trajectory length
        for j in range(l):
          try:
            baseline[j] += rewards[j:].sum()
            ntraj_contribute[j] += 1
          except IndexError:
            baseline.append(rewards[j:].sum())
            ntraj_contribute.append(1) 
      baseline = np.array(baseline)
      ntraj_contribute = np.array(ntraj_contribute)
      baseline /= ntraj_contribute

    for i in range(N):
      obs = trajectories["obs"][i]
      acs = trajectories["acs"][i]
      acs_tensor = torch.tensor(acs, dtype=torch.float, device=self.device)
      rewards = trajectories["rewards"][i]
      rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)

      acs_distribution = self.get_action_distribution(obs)
      log_probs = acs_distribution.log_prob(acs_tensor)
      
      l = obs.shape[0] # Trajectory length
      if self.variance_reduction == "NONE":
        loss -= log_probs.sum() * rewards_tensor.sum()

      elif self.variance_reduction == "REWARD_TO_GO":
        for j in range(l):
          rewards_togo_tensor = rewards_tensor[j:]
          loss -= log_probs[j] * rewards_togo_tensor.sum()

      elif self.variance_reduction == "BASELINE":
        loss -= log_probs.sum() * (rewards_tensor.sum() - baseline)

      elif self.variance_reduction == "BOTH":
        for j in range(l):
          rewards_togo_tensor = rewards_tensor[j:]
          loss -= log_probs[j] * (rewards_togo_tensor.sum() - baseline[j])

    loss /= N
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()











