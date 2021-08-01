class DQNActor():
    def __init__(self, critic):
      self.critic = critic

    def get_action(self, ob):
      qa_value = self.critic.qa_value(ob)
      return qa_value.argmax()