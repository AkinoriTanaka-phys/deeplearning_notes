import numpy as np

def step(env, q, Agt, Opt, Model):
    if env.is_solved() or env.t > 500:
        env.reset()
    s = env.get_state(); a = Agt.play()
    s_next, r_next, _, _ = env.step(a)
    Opt.update(s, a, r_next, s_next)
    ### モデル学習
    Model.memorize(s, a, s_next, r_next)
    ### planning プレーをリプレイ
    for _ in range(50):
        S, A, S_next, R_next = Model.simulate()
        Opt.update(S, A, R_next, S_next)

class ReplayModel():
    def __init__(self, MazeEnv):
        self.reset()
        self.Env = MazeEnv
        
    def reset(self):
        self.S2_A2SR = {}
    
    def memorize(self, S, A, S_next, R_next):
        x, y = S
        x_next, y_next = S_next
        if (x, y) not in list(self.S2_A2SR.keys()):
            self.S2_A2SR.setdefault((x, y), {})
            
        self.S2_A2SR[(x, y)][A] = (x_next, y_next, R_next)
        # ここはデフォルトセットにしてはいけない（後で振る舞いが変わるかもしれないので、そのときは後を優先して上書き）
        
    def simulate(self):
        '''
        1: sample S from memory
        2: sample A from A(S) in memory
        3: get S_next, R_next from S, A
        return: S, A, S_next, R_next
        '''
        xys = list(self.S2_A2SR.keys())
        x, y = xys[np.random.randint(len(xys))]
        As = list(self.S2_A2SR[(x, y)].keys())
        A = As[np.random.randint(len(As))]
        
        x_next, y_next, R_next = self.S2_A2SR[(x, y)][A]
        return np.array([x, y]), A, np.array([x_next, y_next]), R_next

class ReplayModel_plus(ReplayModel):
    def __init__(self, MazeEnv, kappa=1e-3):
        self.reset()
        self.Env = MazeEnv
        self.kappa = kappa
        self.t = 0
    
    def memorize(self, S, A, S_next, R_next):
        #print("m", self.Env.t)
        x, y = S
        x_next, y_next = S_next
        if (x, y) not in list(self.S2_A2SR.keys()):
            self.S2_A2SR.setdefault((x, y), {0:(x,y,0,1), 1:(x,y,0,1), 2:(x,y,0,1), 3:(x,y,0,1)})
            # 一旦全ての行動を起こしたことにしないとプランニングで意味のない行動（したところのある行動）しかしなくなり、
            # 振動などしてむしろ悪くなる
            
        self.S2_A2SR[(x, y)][A] = (x_next, y_next, R_next, self.t)
        self.t += 1
        # 最後に self.t を足した
        
    def simulate(self):
        '''
        1: sample S from memory
        2: sample A from A(S) in memory
        3: get S_next, R_next from S, A
        return: S, A, S_next, R_next
        '''
        xys = list(self.S2_A2SR.keys())
        x, y = xys[np.random.randint(len(xys))]
        As = list(self.S2_A2SR[(x, y)].keys())
        A = As[np.random.randint(len(As))]
        
        x_next, y_next, R_next, t = self.S2_A2SR[(x, y)][A]
        # t を受け取って、報酬を変形
        return np.array([x, y]), A, np.array([x_next, y_next]), R_next+self.kappa*(self.t - t)**(1/2)
