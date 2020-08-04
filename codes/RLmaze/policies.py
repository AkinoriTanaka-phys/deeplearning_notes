import numpy as np

def softmax(xs):
    sps = xs.shape
    num = np.exp(xs)
    den = np.sum(num, axis=2).reshape(sps[0], sps[1], 1)
    return num/den

class Action_value():
    def __init__(self, Env, init=0.01):
        self.values_table = init*np.random.rand(Env.lx*Env.ly*len(Env.action_space)).reshape(Env.lx, Env.ly, len(Env.action_space))

    def get_values(self, s):
        """
        座標＝sでの[Q(s, a=0), Q(s, a=1), Q(s, a=2), Q(s, a=3), ...]
                      を返す
        """
        x, y = s # state
        return self.values_table[x, y, :]

class Parameters():
    def __init__(self, Env, init=0.01):
        self.values_table = init*np.random.rand(Env.lx*Env.ly*4).reshape(Env.lx, Env.ly, 4)
        
    def get_values(self, s):
        """
        座標＝sでの[Q(s, a=0), Q(s, a=1), Q(s, a=2), Q(s, a=3)]
                      を返す
        """
        x, y = s # state
        return self.values_table[x, y, :]

class Policy():
    def __init__(self):
        pass
    
    def sample(self):
        """
        return a number in [0,1,2,3] corresponding to [up, down, left, right]
        """
        action = None
        return action
    
class Random(Policy):
    def __init__(self, Env):
        self.A = Env.action_space # 迷路だと[0, 1, 2, 3]
        
    def sample(self):
        return np.random.choice(self.A) # np.random.choice(リスト)：リストから1つの要素をランダムサンプル
    
class Greedy(Policy):
    def __init__(self, Env, Q):
        self.Q = Q
        self.Env = Env
        
    def returns_action_from(self, values):
        action = np.argmax(values)
        return action
        
    def sample(self):
        Qvalues = self.Q.get_values(self.Env.state)
        return self.returns_action_from(Qvalues)

class EpsilonGreedy(Greedy):
    def __init__(self, Env, Q, epsilon=0.1):
        super(EpsilonGreedy, self).__init__(Env, Q) # python 3 : super().__init__(Env, Q)で十分
        self.epsilon = epsilon
    
    #def returns_action_from(self, values):
    #    if np.random.rand()<1-self.epsilon:
    #        action = np.argmax(values)
    #    else:
    #        action = np.random.choice(np.arange(len(values)))
    #    return action

    def returns_action_from(self, values):
        Na = len(values)
        if np.random.rand()<1-self.epsilon:
            action = np.random.choice(np.arange(Na)[(values == np.max(values))])
        else:
            action = np.random.choice(np.arange(len(values)))
        return action

class Softmax(Policy):
    def __init__(self, Env, f=None, temp=1):
        self.f = f
        self.Env = Env
        self.temp = temp
        
    def get_prob_table(self):
        fvalues_table = self.f.values_table
        #self.f.get_values(state)
        prob = softmax(fvalues_table/self.temp)
        return prob
    
    def get_prob(self, state):
        x, y = state
        return self.get_prob_table()[x, y, :]
        
    def sample(self):
        prob = self.get_prob(self.Env.state)
        action = np.random.choice(self.Env.action_space, p=prob)
               #p=prob のオプションは確率リストprobに従ってaction_spaceからサンプルする
        return action
