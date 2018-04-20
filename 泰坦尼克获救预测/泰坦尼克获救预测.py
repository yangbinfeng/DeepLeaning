
# coding: utf-8

# In[25]:


from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn import preprocessing

data = pd.read_csv("/Users/yangbinfeng/Downloads/python入门/泰坦尼克获救预测/train.csv")

data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Cabin', 'Embarked']]
data['Age']=data['Age'].fillna(data['Age'].median())
data['Age'] = preprocessing.scale(data['Age'])
data['Fare'] = preprocessing.scale(data['Fare'])
data['Cabin']=pd.factorize(data.Cabin)[0]
data.fillna(0,inplace=True)
#data['Sex']=[1 if x=="male" else 0 for x in data.Sex]
data['male']=np.array(data['Sex']=='male').astype(np.int32)
data['female']=np.array(data['Sex']=='female').astype(np.int32)
del data['Sex']

data['p1']=np.array(data['Pclass']==1).astype(np.int32)
data['p2']=np.array(data['Pclass']==2).astype(np.int32)
data['p3']=np.array(data['Pclass']==3).astype(np.int32)
del data['Pclass']

data['e1']=np.array(data['Embarked']=='S').astype(np.int32)
data['e2']=np.array(data['Embarked']=='C').astype(np.int32)
data['e3']=np.array(data['Embarked']=='Q').astype(np.int32)
del data['Embarked']

data_train = data[['male','female', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'p1','p2','p3','e1','e2','e3']]
data_target = data['Survived'].values.reshape(len(data),1)

inputs=Input(shape=(13,))
x=Dense(13,activation='relu')(inputs)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.5)(x)
predictions=Dense(1,activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(data_train, data_target,
          batch_size=100,epochs = 50,verbose=1,shuffle=True,)


# In[40]:


from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn import preprocessing

data = pd.read_csv("/Users/yangbinfeng/Downloads/python入门/泰坦尼克获救预测/train.csv")

data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Cabin', 'Embarked']]
data['Age']=data['Age'].fillna(data['Age'].median())
data['Age'] = preprocessing.scale(data['Age'])
data['Fare'] = preprocessing.scale(data['Fare'])
data['Cabin']=pd.factorize(data.Cabin)[0]
data.fillna(0,inplace=True)
#data['Sex']=[1 if x=="male" else 0 for x in data.Sex]
data['male']=np.array(data['Sex']=='male').astype(np.int32)
data['female']=np.array(data['Sex']=='female').astype(np.int32)
del data['Sex']

data['p1']=np.array(data['Pclass']==1).astype(np.int32)
data['p2']=np.array(data['Pclass']==2).astype(np.int32)
data['p3']=np.array(data['Pclass']==3).astype(np.int32)
del data['Pclass']

data['e1']=np.array(data['Embarked']=='S').astype(np.int32)
data['e2']=np.array(data['Embarked']=='C').astype(np.int32)
data['e3']=np.array(data['Embarked']=='Q').astype(np.int32)
del data['Embarked']

data_train = data[['male','female', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'p1','p2','p3','e1','e2','e3']]
data_target = data['Survived'].values.reshape(len(data),1)

inputs=Input(shape=(13,))
x=Dense(13,activation='relu')(inputs)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.5)(x)
predictions=Dense(1,activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(data_train, data_target,
          batch_size=100,epochs =150,verbose=1,shuffle=True,)

