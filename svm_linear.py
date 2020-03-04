import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics as skm
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Importacion de datos
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x, x_conf, y, y_conf = train_test_split(data, target, train_size=0.8)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_conf = scaler.transform(x_conf)

def PCA(x):
	cov = np.cov(x.T)
	valores, vectores = np.linalg.eig(cov)
	valores = np.real(valores)
	vectores = np.real(vectores)
	ii = np.argsort(-valores)
	valores = valores[ii]
	vectores = vectores[:,ii]

	return vectores

vectores = PCA(x_train)
x_train = x_train @ vectores
x_test = x_test @ vectores
x_conf = x_conf @ vectores

def SVC(c,x_fit,x,y_fit,y):

	svm_ = svm.SVC(C = c)
	svm_.fit(x_fit[:,0:10],y_fit)

	f1 = skm.f1_score(y,svm_.predict(x[:,0:10]), average = 'macro')

	return f1

c = np.logspace(-4,2,30)

f1_c = []

for element in c:
	f1_c.append(SVC(element,x_train,x_test,y_train,y_test))

max_ = np.argmax(f1_c)
best_c = c[max_]

best_svm = svm.SVC(C = best_c)
best_svm.fit(x_train[:,0:10],y_train)
confusion = skm.confusion_matrix(y_conf,best_svm.predict(x_conf[:,0:10]))

fig, ax = plt.subplots()
ax.matshow(confusion)
for i in range(0,10):
    for j in range(0,10):
        c = confusion[j,i]/len(y_conf[y_conf == j])
        ax.text(i, j,r'{:.2f}'.format(c), va='center', ha='center')
plt.title('C = {:.2f}'.format(best_c))
plt.xlabel('Predict')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')