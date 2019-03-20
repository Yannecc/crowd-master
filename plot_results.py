import numpy as np
import matplotlib.pyplot as plt

i = 5 #base shape
max = 5

crowdeds = [[i], [i, i, i], [i, i, i, i, i]]
uncrowdeds = []
for j in range(1, max + 1):
    if j != i:
        uncrowdeds += [[i, j, i, j, i]]

# load results
LOGDIR = 'crowd-master1_logdir/version_0_hidden_512'

vernier = np.squeeze(np.load(LOGDIR+'/vernier_and_shapes_percent_correct[].npy'))
crow = [[],[],[]]
for t in range (3):
    crow[t] = np.squeeze(np.load(LOGDIR+'/vernier_and_shapes_percent_correct' + str(crowdeds[t]) + '.npy'))

uncrow = [[], [], [], []]
for t in range(4):
    uncrow[t] = np.squeeze(np.load(LOGDIR + '/vernier_and_shapes_percent_correct' + str(uncrowdeds[t]) + '.npy'))


# cosmetics
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']


####### PLOT RESULTS #######

N = len(layers)
ind = np.arange(N)  # the x locations for the groups
width = 0.11        # the width of the bars

#increase image size
fig, ax = plt.subplots()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test2png.png', dpi=100)

#Define bars of histogram
rects1 = ax.bar(ind, vernier, width, color=(146./255, 181./255, 88./255))

rects21 = ax.bar(ind + width, crow[0], width, color=(220./255, 76./255, 70./255))
rects22 = ax.bar(ind + 2*width, crow[1], width, color=(235./255, 76./255, 70./255))
rects23 = ax.bar(ind + 3*width, crow[2], width, color=(250./255, 76./255, 70./255))

rects31 = ax.bar(ind + 4*width, uncrow[0], width, color=(79./255, 132./255, 196./255))
rects32 = ax.bar(ind + 5*width, uncrow[1], width, color=(79./255, 142./255, 206./255))
rects33 = ax.bar(ind + 6*width, uncrow[2], width, color=(79./255, 152./255, 216./255))
rects34 = ax.bar(ind + 7*width, uncrow[3], width, color=(79./255, 162./255, 226./255))

# add some text for labels, title and axes ticks, and save figure
ax.set_ylabel('Percent correct')
ax.set_title('Vernier decoding from alexnet layers')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(layers)
ax.plot([-0.3,N], [50, 50], 'r--') # chance level cashed line
ax.legend((rects1[0], rects21[0], rects31[0]), ('vernier', 'crowding 1-3-5', 'uncrowding, 5 shapes'))


plt.savefig(LOGDIR+'/_plots/00Shape'+str(i)+'_plot.png')
plt.ylim(30,100)
plt.savefig(LOGDIR+'/_plots/30Shape'+str(i)+'_plot.png')