# python standard library
import time

# from pypi
import numpy
import pandas

class Actions(object):
    actions = ['Left', 'Right']
    move_left = 'Left'
    move_right = 'Right'

class States(object):
    goal = "terminal"

class QLearner(object):
    """Runs the reinforcement learning

    Args:
     episodes: Number of time to repeat training
     start_state (int): where to start the agent
     states (int): total size for the environment
     terminal (int): where the goal state is
    """
    def __init__(self, episodes=15, start_state=0, states=6, terminal=None):
        self._episodes = None
        self.episodes = episodes
        self.start_state = start_state
        self.states = states
        self._terminal = terminal
        self._environment = None
        self._agent = None
        return
    @property
    def episodes(self):
        """the episodes iterator
    
        Returns:
         range: (1, episodes + 1)
        """
        return self._episodes
    
    @episodes.setter
    def episodes(self, episode_count):
        """creates the episodes iterator
    
        Args:
         episode_count(int): number of episodes to train the agent
        """
        self._episodes = range(1, episode_count + 1)
        return

    @property
    def terminal(self):
        """Goal state
    
        Returns:
         int: zero-based index of the goal-state
        """
        if self._terminal is None:
            self._terminal = self.states - 1
        return self._terminal

    @property
    def environment(self):
        """The Environment for the agent
    
        Returns:
         Environment: one-dimensional environment
        """
        if self._environment is None:
            self._environment = Environment(self.start_state,
                                            self.states,
                                            self.terminal)
        return self._environment

    @property
    def agent(self):
        """The agent that explores the environment
    
        Returns:
         Agent: agent built for the environment
        """
        if self._agent is None:
            self._agent = Agent(self.environment)
        return self._agent

    def __call__(self):
        """runs the episodes to train the agent in the environment
    
        """
        for episode in self.episodes:
            counter = 0
            self.environment.reset()
            self.environment.update(episode, counter)
            while not self.environment.goal_reached:
                self.agent.act()
                counter += 1
                self.environment.update(episode, counter)
        return

class Environment(object):
    """The environment to explore

    Args:
     start_state(int): where the agent will start
     states (int): the size of the world
     terminal (int): where the target state is
     output_pause (float): seconds to sleep after printing to the screen
    """
    def __init__(self, start_state, states, terminal, output_pause=2):
        self.start_state = start_state
        self.state = start_state
        self.next_state = start_state
        self.states = states
        self.terminal = terminal
        self.output_pause = output_pause
        return
    
    def evaluate(self, action):
        """Checks if the action will lead to the goal
    
        Args:
         action (str): one of the actions to explore the environment
    
        Returns:
         int: 1 if this will lead to the goal, 0 otherwise
        """
        if action == Actions.move_right:
            self.next_state = self.state + 1
        else:
            self.next_state = max(self.state - 1, 0)
        reward = 1 if self.next_state == self.terminal else 0
        return reward

    def update(self, episode, step):
        """Emits the updated environment to the user
        
        also sets the state to the next state
    
        Args:
         episode (int): what episode we're in
         step (int): how long we've been running this episode
        """
        environment = ['-'] * (self.states - 1) + ['T']
        if self.goal_reached:
            print("\nEpisode {}: Total Steps = {}".format(episode, step))
            time.sleep(self.output_pause)
        else:
            environment[self.next_state] = 'O'
            print("{}".format("".join(environment)))
        self.state = self.next_state
        return

    @property
    def goal_reached(self):
        """Checks if the next-state is the goal
    
        Returns:
         bool: True if next-state is the goal
        """
        return self.next_state == self.terminal

    def reset(self):
        """Resets the states to the start state"""
        self.state = self.start_state
        self.next_state = self.start_state
        return

class Agent(object):
    """This is the agent that will learn to find the treasure

    Args:
     environment: The environment to explore
     exploitation_rate: Fraction of the time to exploit (epsilon)
     discount_factor: Discount factor (gamma)
     learning_rate: how much to change the reward (alpha)
    """
    def __init__(self, environment, exploitation_rate=0.9, discount_factor=0.9,
                 learning_rate=0.1):
        self.environment = environment
        self.exploitation_rate = exploitation_rate
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self._q_table = None
        return
  
    @property
    def q_table(self):
        """The Quality Estimate table
    
        Each cell is the quality-estimate for a given state, action pair
    
        Returns:
         DataFrame: rows are states, columns are actions
        """
        if self._q_table is None:
            self._q_table = pandas.DataFrame(
                numpy.zeros((self.environment.states, len(Actions.actions))),
                columns=Actions.actions,
            )
            assert self.q_table.shape == (self.environment.states, len(Actions.actions))
        return self._q_table
        

    @property
    def action(self):
        """Return the next chosen action
        
        Returns:
         str: the next action to take
        """
        # get the row in the q-table matching the current state
        state = self.q_table.iloc[self.environment.state, :]
    
        # only explore if we generate a value over epsilon
        # or none of the actions have a reward
        if numpy.random.uniform() > self.exploitation_rate or state.all() == 0:
            action = numpy.random.choice(Actions.actions)
        else: # exploit
            # get the column-name of the cell with the largest value
            action = state.idxmax()
        return action

    def act(self):
        """Updates the Q-table based on the reward from the last action"""
        action = self.action
        reward = self.environment.evaluate(action)
        prediction = self.q_table.loc[self.environment.state, action]
        if self.environment.goal_reached:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.iloc[self.environment.next_state, :].max()
        self.q_table.loc[self.environment.state, action] += self.learning_rate * (target - prediction)
        return
if __name__ == "__main__":
    learner = QLearner(states=10)
    learner()
    print(learner.agent.q_table)
    print("\nlearned-model with only exploitation set")
    learner.agent.exploitation_rate = 1
    learner.episodes = 1
    learner()
