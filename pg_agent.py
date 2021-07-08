from MLP_policy import *

class REINFORCEAgent():
    def __init__(self, params):
        self.params = params
        self.actor = MLPPolicyREINFORCE(params["actor_params"])

    def train(self, trajectories):
        self.actor.update(trajectories)

class PGAgent():
  def __init__(self, params):
      self.params = params
      self.actor = MLPPolicyPG(params["actor_params"])

  def train(self, trajectories):
      self.actor.update(trajectories)