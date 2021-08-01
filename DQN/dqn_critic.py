import torch
import numpy as np

class DQNCritic():
  def __init__(self, params):
    self.ob_dim = params["ob_dim"]
    self.ac_dim = params["ac_dim"]
    self.n_layers = params["n_layers"]
    self.hidden_layer_size = params["hidden_layer_size"]
    self.learning_rate = params["learning_rate"]
    self.discount_factor = params["discount_factor"]
    
    self.device = torch.device('cpu')

    # Building MLP Q-Nets
    layers = []
    in_size = self.ob_dim
    for _ in range(self.n_layers):
      layers.append(torch.nn.Linear(in_size, self.hidden_layer_size))
      layers.append(torch.nn.ReLU())
      in_size = self.hidden_layer_size
    layers.append(torch.nn.Linear(in_size, self.ac_dim))
    self.q_net = torch.nn.Sequential(*layers).to(self.device)
    self.q_net_target = torch.nn.Sequential(*layers).to(self.device)

    # Adam Optimizer
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), self.learning_rate)

  def update(self, obs, acs, next_obs, rews):
    batch_size = rews.shape[0]

    obs_tensor = torch.tensor(obs, dtype=torch.float, device=self.device)
    acs_tensor = torch.tensor(acss, dtype=torch.float, device=self.device)
    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float, device=self.device)
    rews_tensor = torch.tensor(rews, dtype=torch.float, device=self.device)

    # argmax_q_next_obs (i.e. a') = argmax_a'(Q_phi(s', a'))    
    q_next_obs = self.q_net(next_obs_tensor)
    argmax_q_next_obs =  torch.max( q_next_obs , axis=-1 ).indices
    argmax_q_next_obs = torch.unsqueeze(argmax_q_next_obs, 1)

    # q_target_next_obs_argmax = Q_phi'(s', a')
    q_target_next_obs = self.q_net_target(next_obs_tensor)
    q_target_next_obs_argmax =  torch.gather(q_target_next_obs, 1, argmax_q_next_obs)
    q_target_next_obs_argmax = torch.squeeze(q_target_next_obs_argmax)

    target = rews_tensor + (self.discount_factor * q_target_obs_argmax)
    q_obs = self.q_net(obs_tensor)
    q_obs_acs =  torch.gather(q_obs, 1, np.unsqueeze(acs_tensor, 1))
    q_obs_acs = torch.squeeze(q_obs_acs)

    loss = torch.nn.HuberLoss(q_obs_acs, target)

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 10)
    self.optimizer.step()

  def update_target_network(self):
    for target_param, param in zip( self.q_net_target.parameters(), self.q_net.parameters()):
      target_param.data.copy_(param.data)

  def qa_value(self, ob):
      ob_tensor = torch.tensor(ob, dtype=torch.float, device=self.device)
      qa_value = self.q_net(ob_tensor)
      return qa_value.cpu().detach().numpy()













