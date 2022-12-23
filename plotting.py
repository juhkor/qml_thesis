import matplotlib.pyplot as plt
import numpy as np

simple_img = [[[0,0],[0,0]],
              [[1,0],[0,0]],
              [[0,1],[0,0]],
              [[0,0],[1,0]],
              [[0,0],[0,1]],
              [[1,1],[0,0]],
              [[1,0],[1,0]],
              [[1,0],[0,1]],
              [[0,1],[1,0]],
              [[0,1],[0,1]],
              [[0,0],[1,1]],
              [[1,1],[1,0]],
              [[1,1],[0,1]],
              [[0,1],[1,1]],
              [[1,0],[1,1]],
              [[1,1],[1,1]]]
              
simple_q1 = [[-0.15,  1.,    1.,    1.  ],
             [ 0.15,  1.,    1.,    1.  ],
             [-0.15, -1.,    1.,    1.  ],
             [-0.15, -1.,   -1.,    1.  ],
             [ 0.15,  1.,    1.,   -1.  ],
             [ 0.15, -1.,    1.,    1.  ],
             [ 0.15, -1.,   -1.,    1.  ],
             [-0.15,  1.,    1.,   -1.  ],
             [-0.15,  1.,   -1.,    1.  ],
             [ 0.15, -1.,    1.,   -1.  ],
             [ 0.15, -1.,   -1.,   -1.  ],
             [ 0.15,  1.,   -1.,    1.  ],
             [-0.15, -1.,    1.,   -1.  ],
             [ 0.15,  1.,   -1.,   -1.  ],
             [-0.15, -1.,   -1.,   -1.  ],
             [-0.15,  1.,   -1.,   -1.  ]]
            
simple_q2 = [[0.99],
             [0.99],
             [0.7 ],
             [0.7 ],
             [0.7 ],
             [0.67],
             [0.7 ],
             [0.7 ],
             [0.49],
             [0.49],
             [0.49],
             [0.46],
             [0.46],
             [0.35],
             [0.49],
             [0.31]]
            
simple_q3 = [[0.],
             [1.],
             [0.],
             [0.],
             [1.],
             [1.],
             [1.],
             [2.],
             [0.],
             [1.],
             [1.],
             [1.],
             [2.],
             [1.],
             [2.],
             [2.]]
simple_q1 = np.around(simple_q1, 2)
fig, ax = plt.subplots(4, 4)
j = 0

for row in ax:
    for i in row:
        i.imshow(simple_img[j],cmap=plt.cm.gray, vmin=0,vmax=1)
        i.set_xticklabels([])
        i.set_xticks([])
        i.get_yaxis().set_visible(False)
        i.set_xlabel(f'Kernel 1:\n{list(simple_q1[j])}\nKernel 2: {list(simple_q2[j])}\nKernel 3: {list(simple_q3[j])}',loc='center',size='x-large')
        j += 1

fig.suptitle('Quanvolution and convolution result on simple binary images',size='x-large')
plt.tight_layout()
plt.show()