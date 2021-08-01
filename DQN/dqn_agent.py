import utils
from DQN.dqn_actor import DQNActor
from DQN.dqn_critic import DQNCritic

class DQNAgent():
  def __init__(self, params):
    self.env = params["env"]
    self.critic = DQNCritic(params["critic_params"])
    self.actor = DQNActor(self.critic)
    self.target_network_update_hz = params["target_network_update_hz"]
    self.replay_buffer = utils.ReplayBuffer()

  def step_env(self):
    """
    Step the env and store the transition.
    """
    pass
  
  def train(self):
    """
    1. Sample a mini-batch from replay_buffer
    2. Update the critic
        1. Calculate target q-value (i.e. y_i)
        2. Take grad step to update parameter of the q_net
    3. update q_target_net <-- q_net at target_network_update_hz
    """
    pass