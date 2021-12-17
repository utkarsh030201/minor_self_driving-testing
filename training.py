from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
path = 'myData'
data = importDataInfo(path)

##GRAPHING
data = balanceData(data,display=False)

##PREPARE FOR PREPROCESSING
imagesPath, steerings = loadData(path,data)

##SPLIT THE DATA FOR TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

##CREATING A MODEL

model = createModel()
model.summary()

##TRAINING

history = model.fit(batch.Gen(xTrain,yTrain,10,1),steps_per_epoch=300,epochs=10,
            validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

##SAVING
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,0.2])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show
