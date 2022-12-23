import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
datapoints = 20

def weight(height):
    bmi = np.random.normal(20)
    if bmi < 19:
        return 'red', (height/100) ** 2 * bmi
    return 'blue', (height/100) ** 2 * bmi

def height():
    return random.randint(160,200)
    
x = [175]
y = [75]
l = ['blue']

for i in range(0,datapoints):
    heightVal = height()
    x.append(heightVal)
    label, weightVal = weight(heightVal)
    y.append(weightVal)
    l.append(label)

figure, (ax1, ax2) = plt.subplots(1,2)


ax1.plot(x,y,'bo')
ax1.set_xlabel('Height, cm')
ax1.set_ylabel('Weight, kg')
ax1.set_title('Some data')

x = np.array(x).reshape(-1,1)
y = np.array(y)

model = LinearRegression()
model.fit(x,y)

pred_y = model.predict(x)
print(pred_y[0])

ax2.plot(x,y,'bo')
ax2.plot(x,pred_y)
ax2.set_xlabel('Height, cm')
ax2.set_ylabel('Weight, kg')
ax2.set_title('Linear regression line fitted on data')
ax2.arrow(x=x[0],y=y[0],dx=0,dy=pred_y[0]-y[0]+1, width=.08,head_width=0.5)
ax2.arrow(x=x[0],y=pred_y[0],dx=0,dy=y[0]-pred_y[0]-1, width=.08,head_width=0.5)
ax2.annotate('SE', xy=(x[0], pred_y[0]+5),
            xytext=(x[0]-3, y[0]-5))
plt.show()


center1 = (50, 60)
center2 = (80, 20)
distance = 20


x1 = np.random.uniform(center1[0], center1[0] + distance, size=(100,))
y1 = np.random.normal(center1[1], distance, size=(100,)) 

x2 = np.random.uniform(center2[0], center2[0] + distance, size=(100,))
y2 = np.random.normal(center2[1], distance, size=(100,)) 

plt.scatter(x1, y1,color='blue')
plt.scatter(x2, y2,color='red',marker='+')
plt.show()

plt.scatter(x1, y1,color='blue')
plt.scatter(x2, y2,color='blue')
plt.show()