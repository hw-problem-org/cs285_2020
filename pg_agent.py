from MLP_policy import *

class PGAgent():
  def __init__(self, params):
      self.params = params
      self.actor = MLPPolicyPG(params["actor_params"])

  def train(self, trajectories):
      self.actor.update(trajectories)