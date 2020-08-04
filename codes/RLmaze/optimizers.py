import numpy as np

class Optimizer():
    def __init__(self, Agt):
        self.Agt = Agt
    
    def update(self):
        """
        なんかいい感じの処理
        """
        pass
    
class SARSA_optimizer(Optimizer):
    def __init__(self, Agt, eta, gamma):
        self.Agent = Agt
        self.Q = Agt.Policy.Q
        self.eta = eta
        self.gamma = gamma

    def update(self, s, a, r_next, s_next):
        a_next = self.Agent.play() # 一回プレイさせてa_nextをサンプル
        TD_error = self.Q.get_values(s)[a] - (r_next + 
                                              self.gamma*self.Q.get_values(s_next)[a_next])
        self.Q.get_values(s)[a] -= self.eta*TD_error
        
class Qlearning_optimizer(Optimizer):
    def __init__(self, Agt, eta, gamma):
        self.Agent = Agt
        self.Q = Agt.Policy.Q
        self.eta = eta
        self.gamma = gamma

    def update(self, s, a, r_next, s_next):
        error = self.Q.get_values(s)[a] - (r_next + 
                                           self.gamma*np.max(self.Q.get_values(s_next)))
                                                       # ↑ ここが変わった
        self.Q.get_values(s)[a] -= self.eta*error

class REINFORCE_optimizer(Optimizer):
    def __init__(self, Agt, eta):
        self.Policy = Agt.Policy
        self.f = Agt.Policy.f
        self.Env = Agt.Policy.Env
        self.eta = eta
        self.N_sa = Parameters(self.Env, init=0) # N_{(s,a)}を数えるためParametersを利用
        self.N = 0 # 実際にself.fが更新された回数を数える。なくてもよい
            
    def update(self, s, a, r_next, s_next):
        x, y = s
        if self.Env.is_solved():
            self.N_sa.values_table[x, y, a] +=10 # 最後にボーナス（理論から外れるがこれがないと遅い）
            T = self.Env.t + 0.01 # オーバーフロー対策(たまに偶然ゴールに落とされてしまうので。。。)
            N_sa = self.N_sa.values_table # N_{(s,a)}を読み込む
            N_s = np.sum(N_sa, axis=2).reshape(self.Env.lx, self.Env.ly, 1)
            g = (N_sa - N_s*self.Policy.get_prob_table())/T # 方策勾配
            self.f.values_table += self.eta*g # 更新
            self.N += 1 # なくてもよい
        else:
            self.N_sa.values_table[x, y, a] +=1 # ここは必須、ゴールしてない時はN_{(s,a)}を更新
            
    def reset(self):
        self.N_sa.values_table[:,:,:] = 0*self.N_sa.values_table[:,:,:]
        # エピソードごとにN_{(s,a)}をリセットする。これは下の学習ループで唱える
                    
###### not yet
class ActorCritic_optimizer(Optimizer):
    def __init__(self, Actor, Critic, lrA, lrC, gamma):
        self.f = Actor.f
        self.V = Critic.V
        self.gamma = gamma
        self.lrA = lrA
        self.lrC = lrC
        
    def return_TD_error(self, x, y, xp, yp, reward):
        return reward + self.gamma*self.V[xp,yp] - self.V[x,y]
        
    def update_Critic(self, x, y, xp, yp, reward, action_label):  
        self.V[x,y] += self.lrC*self.return_TD_error(x, y, xp, yp, reward)
            
    def update_Actor(self, x, y, xp, yp, reward, action_label):
        self.f[x, y, action_label] += self.lrA*self.return_TD_error(x, y, xp, yp, reward)
