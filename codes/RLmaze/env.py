import numpy as np

import matplotlib.pyplot as plt
import matplotlib.collections as mc
import copy


action2vect = {0: np.array([0, -1]),
               1: np.array([0, +1]),
               2: np.array([-1, 0]),
               3: np.array([+1, 0])
               }

action2vect2 = {0: np.array([0, -1]),
               1: np.array([0, +1]),
               2: np.array([-1, 0]),
               3: np.array([+1, 0]),
               4: np.array([-1, -1]),
               5: np.array([-1, +1]),
               6: np.array([+1, -1]),
               7: np.array([+1, -1])
               }

action2vect3 = {0: np.array([0, -1]),
               1: np.array([0, +1]),
               2: np.array([-1, 0]),
               3: np.array([+1, 0]),
               4: np.array([-1, -1]),
               5: np.array([-1, +1]),
               6: np.array([+1, -1]),
               7: np.array([+1, -1]),
               8: np.array([0, 0])
               }

a2m = {0:'up', 1:'down', 2:'left', 3:'right'}

def random_initialize(Maze):
    floor_labels = np.arange(len(Maze.floors))
    start_floor_label = np.random.choice(floor_labels)
    goal_floor_label = np.random.choice(floor_labels)
    #Maze.set_start(Maze.floors[start_floor_label].tolist())
    Maze.set_goal(Maze.floors[goal_floor_label].tolist())
    return Maze

def get_fig_ax(size=(8, 5)):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig, ax
        
class MazeEnv():
    def __init__(self, lx, ly, threshold=0.9, figsize=5, const=0):
        self.lx = lx
        self.ly = ly
        self.create_maze_by_normal_distribution(threshold=threshold)
        self = random_initialize(self)
        
        self.action_space = [0,1,2,3]
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const
        self.reward_goal = 1
        self.reward_usual = 0
    def reset(self, coordinate=[None, None]):
        """
        put the state at the start.
        """
        if coordinate[0]!=None:
            self.state = np.array(coordinate)
        else:
            #
            floor_labels = np.arange(len(self.floors))
            start_floor_label = np.random.choice(floor_labels)
            self.state = self.floors[start_floor_label]
            #
            #self.state = np.array(self.start)
        self.status = 'Reset'
        self.t = 0
        return self.get_state()
        
    def is_solved(self):
        """
        if the state is at the goal, returns True.
        """
        return self.goal==self.state.tolist()
    
    def get_state(self):
        """
        returns (x, y) coordinate of the state
        """
        return copy.deepcopy(self.state)#, copy.deepcopy(self.state[1])
            
    def step0(self, state, action):
        add_vector_np = action2vect[action]
        if (state+add_vector_np).tolist() in self.floors.tolist():
            next_state = state+add_vector_np
            self.status = 'Moved'
        else:
            next_state = state
            self.status = 'Move failed'
        self.t += 1
        return next_state
    
    def step1(self, state, action, state_p):
        if state_p.tolist()==self.goal:
            reward = self.reward_goal
        elif False:
            reward = 0.1
        else:
            reward = self.reward_usual
        return reward + self.reward_const
    
    def step(self, action):
        state = self.get_state()
        next_state = self.step0(state, action)
        reward = self.step1(state, action, next_state)
        # self.state update
        self.state = next_state
        return self.get_state(), reward, self.is_solved(), {}
        
    def create_maze_by_normal_distribution(self, threshold):
        """
        creating a random maze.
        Higher threshold creates easier maze.
        around threshold=1 is recomended.
        """
        x = np.random.randn(self.lx*self.ly).reshape(self.lx, self.ly)
        y = (x < threshold)*(x > -threshold)
        self.tile = y
        self.load_tile()
        
    def load_tile(self):
        self.floors = np.array(list(np.where(self.tile==True))).T # (#white tiles, 2), 2 means (x,y) coordinate
        self.holes = np.array(list(np.where(self.tile==True))).T # (#black tiles, 2)

    def flip(self, coordinate=[None, None]):
        self.tile[coordinate[0], coordinate[1]] = not self.tile[coordinate[0], coordinate[1]]
        self.load_tile()
    
    def render_tile(self, ax, cmap='gray'):
        #ax.imshow(self.tile.T, interpolation="none", cmap=cmap)
        ax.imshow(self.tile.T, interpolation="none", cmap=cmap, vmin=0, vmax=1)
        return ax
    
    def render_arrows(self, ax, values_table):
        lx, ly, _ = values_table.shape
        #vmaxs = np.max(values_table, axis=2).reshape(lx, ly, 1)
        vmins = np.min(values_table, axis=2).reshape(lx, ly, 1)
        offset = - vmins
        vnoed = values_table*self.tile.reshape(lx, ly, 1) + offset 
        vnoed_maxs = np.max(vnoed, axis=2).reshape(lx, ly, 1)
        #vt = np.transpose(values_table*self.tile.reshape(lx, ly, 1)/vmaxs, (1,0,2))
        vt = np.transpose(vnoed/(vnoed_maxs), (1,0,2))
        #print(vt)
        width = 0.5
        X, Y= np.meshgrid(np.arange(0, lx, 1), np.arange(0, ly, 1))
        ones = .5*np.ones(lx*ly).reshape(lx, ly)
        zeros= np.zeros(lx*ly).reshape(lx, ly)
        # up
        ax.quiver(X, Y, zeros, ones, vt[:,:,0], alpha=0.8, 
                      cmap='Reds', scale_units='xy', scale=1)
        # down
        ax.quiver(X, Y, zeros, -ones, vt[:,:,1], alpha=0.8, 
                      cmap='Reds', scale_units='xy', scale=1)
        # left
        ax.quiver(X, Y, -ones, zeros, vt[:,:,2], alpha=0.8, 
                      cmap='Reds', scale_units='xy', scale=1)
        # right
        ax.quiver(X, Y, ones, zeros, vt[:,:,3], alpha=0.8, 
                      cmap='Reds', scale_units='xy', scale=1)
        return ax
        
    def render(self, fig=None, ax=None, lines=None, values_table=None,
               linecolor='black', canvas=False, interactive=False):
        if interactive:
            #pass
            #canvas = True
            ax.clear()
        elif ax is not None:
            pass
        else:
            fig = plt.figure(figsize=(self.figsize, self.figsize))
            ax = fig.add_subplot(111)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        ####
        ax = self.render_tile(ax)
        
        if values_table is not None:
            ax = self.render_arrows(ax, values_table)
        ####
        try:
            ax.scatter(self.start[0], self.start[1], marker='x', s=100, color='blue',
                       alpha=0.8, label='start')
        except AttributeError:
            pass
        try:
            ax.scatter(self.goal[0], self.goal[1], marker='d', s=100, color='red',
                       alpha=0.8, label='goal')
        except AttributeError:
            pass
        try:
            ax.scatter(self.state[0], self.state[1], marker='o', s=100, color='black',
                       alpha=0.8, label='agent')
        except (AttributeError, TypeError):
            pass
        if lines is not None:
            lc = mc.LineCollection(lines, linewidths=2, color=linecolor, alpha=0.5)
            ax.add_collection(lc)
        else:
            pass
            
        if interactive:
            #print('interactive')
            ax.figure.canvas.draw()
            return ax
        elif canvas:
            #print('canvas')
            #ax.figure.canvas.draw()
            return ax
        else:
            #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', scatterpoints=1)
            plt.show()
        
    def set_start(self, coordinate=[None, None]):
        if coordinate in self.floors.tolist():
            self.start = coordinate
        else:
            print('Set the start on a white tile.')
            
    def set_goal(self, coordinate=[None, None]):
        if coordinate in self.floors.tolist():
            self.goal = coordinate
        else:
            print('Set the goal on a white tile.')
                      
    def play(self, Agent, show=True, fig=None, ax=None, linecolor='black', canvas=False):
        lines = []
        while not self.is_solved():
            state0 = self.get_state()
            action = Agent.play()
            self.step(action)
            state1 = self.get_state()
            lines.append([state0, state1])
        if show:
            return self.render(fig=fig, ax=ax, lines=lines, linecolor=linecolor, canvas=canvas, interactive=False)
                
    def play_interactive(self, Agent):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        self.render(fig=fig, ax=ax)
        lines = []
        while not self.is_solved():
            state0 = self.get_state()
            action = Agent.play()
            self.step(action)
            state1 = self.get_state()
            lines.append([state0, state1])
            self.render(fig=fig, ax=ax, lines=lines, interactive=True)
            #fig.canvas.draw()
        self.render(fig=fig, ax=ax, lines=lines)
        plt.show()
        print("solved!")


class CliffEnv(MazeEnv):
    def __init__(self, lx, ly, threshold=0.9, figsize=5, const=0):
        self.lx = lx
        self.ly = ly
        self.create_cliff()
        self.start = [0, ly-1]
        self.goal = [lx-1, ly-1]
        
        self.action_space = [0,1,2,3]
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const
        self.reward_goal = 0
        self.reward_usual = -1
        
    def reset(self, coordinate=[None, None]):
        """
        put the state at the start.
        """
        if coordinate[0]!=None:
            self.state = np.array(coordinate)
        else:
            self.state = np.array(self.start)
        self.status = 'Reset'
        self.t = 0
        return self.get_state()
    
    def create_cliff(self):
        """
        creating a cliff
        """
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        x[:, self.ly-1] -= 1
        x[0, self.ly-1] += 1
        x[self.lx-1, self.ly-1] += 1
        self.tile = x
        self.load_tile()
        
    def render_tile(self, ax, cmap='Reds_r'):
        ax.imshow(self.tile.T, interpolation="none", cmap=cmap)
        return ax
    
    def step0(self, state, action):
        add_vector_np = action2vect[action]
        if (state+add_vector_np).tolist() in self.floors.tolist():
            next_state = state+add_vector_np
            self.status = 'Moved'
        elif (state+add_vector_np).tolist() in self.holes.tolist():
            next_state = self.start
            self.status = 'Dropped'
        else:
            next_state = state
            self.status = 'Move failed'
        self.t += 1
        return next_state
    
    def step1(self, state, action, state_p):
        if state_p.tolist()==self.goal:
            reward = self.reward_goal
        elif self.status=='Dropped':
            reward = -100
        else:
            reward = self.reward_usual
        return reward

class GridWorldEnv(MazeEnv):
    def __init__(self, lx=5, ly=5, threshold=None, figsize=5,
                 A=[1,0], Ap=[1,4], B=[3,0], Bp=[3,2], const=0):
        self.lx = lx
        self.ly = ly
        self.A = A
        self.Ap= Ap
        self.B = B
        self.Bp=Bp

        self.create_world()
        
        self.action_space = [0,1,2,3]
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const

    def is_solved(self):
        return False

    def create_world(self):
        """
        creating world without any wall
        """
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        #x[0,:] = 0
        #x[self.lx-1,:] = 0
        #x[:,0] = 0
        #x[:,self.ly-1] = 0
        self.tile = x
        self.load_tile()

    def step0(self, state, action):
        add_vector_np = action2vect[action]
        s_next_candidate = (state+add_vector_np).tolist()
        if self.get_state().tolist() == self.A:
            next_state = np.array(self.Ap)
            self.status = 'From A'
        elif self.get_state().tolist() == self.B:
            next_state = np.array(self.Bp)
            self.status = 'From B'
        elif s_next_candidate in self.floors.tolist():
            next_state = state+add_vector_np
            self.status = 'Moved'
        else: #s_next_candidate in self.holes.tolist():
            next_state = state
            self.status = 'Off grid'
        self.t += 1
        return next_state

    def step1(self, state, action, state_p):
        if self.status=='From A':
            reward = 10
        elif self.status=='From B':
            reward = 5
        elif self.status=='Off grid':
            reward = -1
        else:
            reward = 0
        return reward + self.reward_const

    def render_tile(self, ax, cmap='Reds_r'):
        ax.imshow(self.tile.T, interpolation="none", cmap=cmap, vmin=0, vmax=1)
        try:
            plt.text(self.A[0], self.A[1], 'A', fontsize=20, ha='center', va='center')
            plt.text(self.Ap[0], self.Ap[1], 'A\'', fontsize=20, ha='center', va='center')
            plt.text(self.B[0], self.B[1], 'B', fontsize=20, ha='center', va='center')
            plt.text(self.Bp[0], self.Bp[1], 'B\'', fontsize=20, ha='center', va='center')
        except AttributeError:
            pass
        return ax

class LongMazeEnv(MazeEnv):
    def __init__(self, ly, threshold=0.9, figsize=5, const=0):
        self.ly = ly
        self.lx = 3*ly
        self.create_maze()
        #self = random_initialize(self)
        self.goal = [self.lx-1, self.ly//2]
        self.start = [0, 0]

        self.action_space = [0,1,2,3]
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const
        self.reward_goal = 1
        self.reward_usual = 0

    def create_maze(self):
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        x[self.lx//3, :] = 0
        x[2*self.lx//3, :] = 0
        x[:, 0] = 1
        x[:, self.ly-1] = 1
        #x[self.lx//2+2, self.ly-1] = 0
        self.tile = x
        self.load_tile()

class GridWorld4x4Env(GridWorldEnv):
    def __init__(self, figsize=5, const=-1):
        self.ly = 4
        self.lx = 4
        self.create_maze()
        self.goal1 = [0, 0]
        self.goal2 = [3, 3]

        self.action_space = [0,1,2,3]
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const
        self.reward_goal = 0
        self.reward_usual = 0

    def create_maze(self):
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        self.tile = x
        self.load_tile()

    def is_solved(self):
        """
        if the state is at the goal, returns True.
        """
        return self.goal1==self.state.tolist() or self.goal2==self.state.tolist()

    def render_tile(self, ax, cmap='Reds_r'):
        ax.scatter(self.goal1[0], self.goal1[1], marker='d', s=100, color='red',
                       alpha=0.8, label='goal1')
        ax.scatter(self.goal2[0], self.goal2[1], marker='d', s=100, color='red',
                       alpha=0.8, label='goal2')
        return super().render_tile(ax, cmap)

    def step0(self, state, action):
        add_vector_np = action2vect[action]
        s_next_candidate = (state+add_vector_np).tolist()
        if s_next_candidate == self.goal1 or s_next_candidate == self.goal2:
            next_state = state+add_vector_np
            self.status = 'goal'
        elif s_next_candidate in self.floors.tolist():
            next_state = state+add_vector_np
            self.status = 'Moved'
        else: #s_next_candidate in self.holes.tolist():
            next_state = state
            self.status = 'Off grid'
        self.t += 1
        return next_state

    def step1(self, state, action, state_p):
        if self.status=='goal':
            reward = self.reward_goal
        else:
            reward = self.reward_usual
        return reward + self.reward_const

class WindyGridworld(MazeEnv):
    def __init__(self, threshold=0.9, figsize=5, const=0):
        self.ly = 7
        self.lx = 10
        self.create_maze()
        self.goal = [7, 3]
        self.start = [0, 3]

        self.windy1_x = [3,4,5,8]
        self.windy2_x = [6,7]

        self.ones = np.zeros(self.lx*self.ly).reshape(self.lx, self.ly)
        for x in range(self.lx):
            if x in self.windy1_x:
                v = 1
            elif x in self.windy2_x:
                v = 2.0
            else:
                v = 0
            for y in range(self.ly):
                if (x in self.windy2_x) and (y in [1,3,5,7]):
                    self.ones[x,y] = 0
                else:
                    self.ones[x,y] = v
            
        self.zeros = np.zeros(self.lx*self.ly).reshape(self.lx, self.ly)
            

        self.action_space = [0,1,2,3]
        self.action2vect = action2vect
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const
        self.reward_goal = 0
        self.reward_usual = -1

    def reset(self, coordinate=[None, None]):
        if coordinate[0]!=None:
            self.state = np.array(coordinate)
        else:
            self.state = np.array(self.start)
        self.status = 'Reset'
        self.t = 0
        return self.get_state()

    def create_maze(self):
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        self.tile = x
        self.load_tile()

    def sample_wind(self, deg):
        return [0, -deg]

    def step0(self, state, action):
        add_vector_np = self.action2vect[action]
        if state[0] in self.windy1_x:
            wind = self.sample_wind(deg=1)
            if (state+wind).tolist() in self.floors.tolist():
                next_state = state + wind
                self.status = 'Moved by wind'
            else:
                next_state = state
        elif state[0] in self.windy2_x:
            wind = self.sample_wind(deg=2)
            wind2 = self.sample_wind(deg=1)
            if (state+wind).tolist() in self.floors.tolist():
                next_state  = state + wind
                self.status = 'Moved by wind'
            elif (state+wind2).tolist() in self.floors.tolist():
                next_state  = state + wind2
                self.status = 'Moved by wind'
            else:
                next_state = state
        else:
            next_state = state

        if (next_state+add_vector_np).tolist() in self.floors.tolist():
            next_state += add_vector_np
            self.status = 'Moved'
        else:
            self.status = 'Move failed'

        self.t += 1
        return next_state

    def render_tile(self, ax, cmap='gray'):
        ax.imshow(self.tile.T, interpolation="none", cmap=cmap, vmin=0, vmax=1)
        #vnoed = values_table*self.tile.reshape(lx, ly, 1) + offset
        #vnoed_maxs = np.max(vnoed, axis=2).reshape(lx, ly, 1)
        #vt = np.transpose(values_table*self.tile.reshape(lx, ly, 1)/vmaxs, (1,0,2))
        #vt = np.transpose(, (1,0,2))
        #print(vt)
        width = 0.5
        X, Y= np.meshgrid(np.arange(0, self.lx, 1), np.arange(0, self.ly, 1))
        # up
        ax.quiver(X, Y, self.zeros.T, self.ones.T, alpha=0.5,
                      color='blue', scale_units='xy', scale=1)
        return ax

class WindyGridworldKingmove(WindyGridworld):
    def __init__(self, wait_is_included=False):
        super().__init__()
        if wait_is_included:
            self.action_space = [0,1,2,3,4,5,6,7,8]
            self.action2vect = action2vect3
        else:
            self.action_space = [0,1,2,3,4,5,6,7]
            self.action2vect = action2vect2

class MaxBiasWalk(MazeEnv):
    def __init__(self, figsize=5, const=0):
        self.ly = 1
        self.lx = 4
        self.create_maze()
        self.goal1 = [0, 0]
        self.goal2 = [3, 0]
        self.start = [2, 0]

        self.action_space = [2,3]
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const
        self.reward_goal = 0
        self.reward_usual = 0

    def reset(self, coordinate=[None, None]):
        if coordinate[0]!=None:
            self.state = np.array(coordinate)
        else:
            self.state = np.array(self.start)
        self.status = 'Reset'
        self.t = 0
        return self.get_state()

    def create_maze(self):
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        self.tile = x
        self.load_tile()

    def is_solved(self):
        """
        if the state is at the goal, returns True.
        """
        return self.status=="bad goal" or self.status=="goal"

    def render_tile(self, ax, cmap='Blues_r'):
        ax.scatter(self.goal1[0], self.goal1[1], marker='d', s=100, color='black',
                       alpha=0.8, label='r~N(-0.1, 1)')
        ax.scatter(self.goal2[0], self.goal2[1], marker='d', s=100, color='red',
                       alpha=0.8, label='r=0')
        return super().render_tile(ax, cmap)

    def render_arrows(self, ax, values_table):
        lx, ly, _ = values_table.shape
        #vmaxs = np.max(values_table, axis=2).reshape(lx, ly, 1)
        vmins = np.min(values_table, axis=2).reshape(lx, ly, 1)
        offset = - vmins
        vnoed = values_table*self.tile.reshape(lx, ly, 1) + offset
        vnoed_maxs = np.max(vnoed, axis=2).reshape(lx, ly, 1)
        #vt = np.transpose(values_table*self.tile.reshape(lx, ly, 1)/vmaxs, (1,0,2))
        vt = np.transpose(vnoed/(vnoed_maxs), (1,0,2))
        #print(vt)
        width = 0.5
        X, Y= np.meshgrid(np.arange(0, lx, 1), np.arange(0, ly, 1))
        ones = .5*np.ones(lx*ly).reshape(lx, ly)
        zeros= np.zeros(lx*ly).reshape(lx, ly)
        # left
        ax.quiver(X, Y, -ones, zeros, vt[:,:,0], alpha=0.8,
                      cmap='Reds', scale_units='xy', scale=1)
        # right
        ax.quiver(X, Y, ones, zeros, vt[:,:,1], alpha=0.8,
                      cmap='Reds', scale_units='xy', scale=1)
        return ax

    def step0(self, state, action):
        if state.tolist() == [1,0]:
            next_state = np.array([0,0])
            self.status = 'bad goal'
        else:
            add_vector_np = action2vect[action+2]
            s_next_candidate = (state+add_vector_np).tolist()
            if s_next_candidate == self.goal2:
            	next_state = state+add_vector_np
            	self.status = 'goal'
            elif s_next_candidate in self.floors.tolist():
            	next_state = state+add_vector_np
            	self.status = 'Moved'
            else: #s_next_candidate in self.holes.tolist():
            	next_state = state
            	self.status = 'Off grid'
        self.t += 1
        return next_state

    def step1(self, state, action, state_p):
        if self.status=='bad goal':
            reward = np.random.normal(-0.1, 1)
        elif self.status=='goal':
            reward = self.reward_goal
        else:
            reward = self.reward_usual
        return reward + self.reward_const

class DynaMaze(MazeEnv):
    def __init__(self, figsize=5, const=0):
        self.ly = 6
        self.lx = 9
        self.create_maze()
        self.goal = [8, 0]
        self.start = [0, 2]

        self.action_space = [0,1,2,3]
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const
        self.reward_goal = 1
        self.reward_usual = 0

    def reset(self, coordinate=[None, None]):
        if coordinate[0]!=None:
            self.state = np.array(coordinate)
        else:
            self.state = np.array(self.start)
        self.status = 'Reset'
        self.t = 0
        return self.get_state()

    def create_maze(self):
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        x[2,1] = 0
        x[2,2] = 0
        x[2,3] = 0

        x[5,4] = 0

        x[7,0] = 0
        x[7,1] = 0
        x[7,2] = 0
        self.tile = x
        self.load_tile()

class BlockingMaze(MazeEnv):
    def __init__(self, figsize=5, const=0):
        self.ly = 6
        self.lx = 9
        self.t = 0
        self.create_maze()
        self.goal = [8, 0]
        self.start = [3, 5]

        self.action_space = [0,1,2,3]
        self.status = 'Initialized'
        self.figsize = figsize

        self.reward_const = const
        self.reward_goal = 1
        self.reward_usual = 0

    def reset(self, coordinate=[None, None], q=None):
        self.create_maze()
        if coordinate[0]!=None:
            self.state = np.array(coordinate)
        else:
            self.state = np.array(self.start)
        self.status = 'Reset'
        #self.t = 0
        
        return self.get_state()

    def create_maze(self):
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        x[:,3] = 0
        if self.t<1000:
            x[8, 3] = 1
        else:
            x[0, 3] = 1        
        self.tile = x
        self.load_tile()

class ShortcutMaze(BlockingMaze):
    def create_maze(self):
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        x[:,3] = 0
        x[0, 3] = 1
        if self.t<3000:
            pass
        else:
            x[8, 3] = 1 
        self.tile = x
        self.load_tile()

class ShortcutMaze2(BlockingMaze):
    def create_maze(self):
        x = np.ones(self.lx*self.ly).reshape(self.lx, self.ly)
        x[:,3] = 0
        x[8, 3] = 1
        if self.t<3000:
            pass
        else:
            x[0, 3] = 1
        self.tile = x
        self.load_tile()

class RandomWalk():
    def __init__(self):
        self.states = ['terminal 1','A', 'B', 'C', 'D', 'E', 'terminal 2']
        self.actions= ['left', 'right']
        self.reset()
    
    def reset(self):
        self.state = 'C'
        self.t = 0
        
    def get_state(self):
        return copy.deepcopy(self.state)
    
    def step0(self, S, A):
        if A=='right':
            S_next = self.states[self.states.index(S)+1]
        else:
            S_next = self.states[self.states.index(S)-1]
        return S_next
        
    def step1(self, S, A, S_next):
        if S_next=='terminal 2':
            return 1
        else:
            return 0
        
    def is_finished(self):
        return self.get_state()=='terminal 1' or self.get_state()=='terminal 2'
        
    def step(self, A):
        S = self.get_state()
        S_next = self.step0(S, A)
        R_next = self.step1(S, A, S_next)
        self.state = S_next
        self.t += 1
        return S_next, R_next, self.is_finished(), {}
    
    def render(self, ax=None, V=None):
        p = False
        if ax==None:
            fig = plt.figure(figsize=(10, 2))
            ax = fig.add_subplot(111)
            p = True
        graph = np.ones(7).reshape(1, 7)
        graph[:,0] = 0
        graph[:,6] = 0
        x = self.states.index(self.state)
        ax.imshow(graph, interpolation="none", cmap='gray')
        ax.scatter(x, 0, marker='o', s=400, color='black',
                       alpha=0.5, label='agt')
        ax.text(1, 0, 'A', fontsize=20, ha='center', va='center', color='red')
        ax.text(2, 0, 'B', fontsize=20, ha='center', va='center', color='red')
        ax.text(3, 0, 'C', fontsize=20, ha='center', va='center', color='red')
        ax.text(4, 0, 'D', fontsize=20, ha='center', va='center', color='red')
        ax.text(5, 0, 'E', fontsize=20, ha='center', va='center', color='red')
        ax.scatter(6, 0, marker='d', s=400, color='blue',
                       alpha=1, label='agt')
        if V is not None:
            ax.imshow(V, interpolation="none", cmap='Blues', alpha=0.5)
        

             
