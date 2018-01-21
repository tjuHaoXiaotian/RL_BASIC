from collections import deque
import numpy as np
import random


'''
flag = tf.app.flags
FLAG = flag.FLAGS
flag.DEFINE_string('size','5','size')
print flag.FLAGS.size
'''

class Experience_replay:
    def __init__(self, size, action_dim,state_dim):
        self.d = deque(maxlen=size)
        self.action_dim = action_dim
        self.state_dim = state_dim

    def experience_in(self, memory):
        self.d.append(memory)

    def experience_out(self, sample_size):
        s_list = random.sample(self.d, sample_size)

        rs = np.asarray([i[0] for i in s_list], dtype=np.float32).reshape((sample_size, self.state_dim))
        ra = np.asarray([i[1] for i in s_list], dtype=np.float32).reshape((sample_size, self.action_dim))
        rr = np.asarray([i[2] for i in s_list], dtype=np.float32).reshape((sample_size, 1))
        rss = np.asarray([i[3] for i in s_list], dtype=np.float32).reshape((sample_size, self.state_dim))
        rt = np.asarray([i[4] for i in s_list], dtype=np.bool).reshape((sample_size, 1))

        return rs, ra, rr, rss, rt

    def experience_out_partly(self,sample_size,part_experience_size):
        sample_index = np.random.randint(0,part_experience_size,sample_size).tolist()

        rs = np.asarray([self.d[i][0] for i in sample_index], dtype=np.float32).reshape((sample_size, self.state_dim))

        return rs


#############test###########
if __name__ == "__main__":
    pass