#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Проверка-данных" data-toc-modified-id="Проверка-данных-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Проверка данных</a></span></li><li><span><a href="#Подготовка-данных-для-машинного-обучения" data-toc-modified-id="Подготовка-данных-для-машинного-обучения-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Подготовка данных для машинного обучения</a></span></li><li><span><a href="#Создаем-учебную,-валидационную-и-тестовые-выборки" data-toc-modified-id="Создаем-учебную,-валидационную-и-тестовые-выборки-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Создаем учебную, валидационную и тестовые выборки</a></span></li><li><span><a href="#Масштабирование" data-toc-modified-id="Масштабирование-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Масштабирование</a></span></li></ul></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span><ul class="toc-item"><li><span><a href="#Обучаем-модели-на-обучающей-выборке-и-находим-точность-каждой-из-них-на-валидационной" data-toc-modified-id="Обучаем-модели-на-обучающей-выборке-и-находим-точность-каждой-из-них-на-валидационной-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Обучаем модели на обучающей выборке и находим точность каждой из них на валидационной</a></span></li><li><span><a href="#Проверка-на-адекватность-моделей" data-toc-modified-id="Проверка-на-адекватность-моделей-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Проверка на адекватность моделей</a></span></li></ul></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span><ul class="toc-item"><li><span><a href="#Для-увеличения-точности-моделей-необходимо-прийти-к-баллансу-классов-данных,-дающих-положительный-и-отрицательные-ответы-в-целевом-признаке" data-toc-modified-id="Для-увеличения-точности-моделей-необходимо-прийти-к-баллансу-классов-данных,-дающих-положительный-и-отрицательные-ответы-в-целевом-признаке-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Для увеличения точности моделей необходимо прийти к баллансу классов данных, дающих положительный и отрицательные ответы в целевом признаке</a></span></li><li><span><a href="#Обучение-моделей-на-сбалансированой-выборке" data-toc-modified-id="Обучение-моделей-на-сбалансированой-выборке-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Обучение моделей на сбалансированой выборке</a></span></li></ul></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li></ul></div>

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте *F1*-меру на тестовой выборке самостоятельно.
# 
# Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# ## Подготовка данных

# ###  Проверка данных 

# In[1]:


# Импортируем необходимые библиотеки для выполнения проекта
import pandas as pd 
import numpy as np 
import seaborn as sns 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


# In[2]:


data= pd.read_csv('/datasets/Churn.csv')


# In[3]:


#Общая информация

print('Общая информация:')

display(data.info())

print(30*' =')

#Вывод первых пяти строк

print('Первые пять строк датафрейма:')

display(data.head())

print(30*' =')

print('Название столбцов датафрейма:')

print(f'Название столбцов: {list(data.columns)}')

print(30*' =')

#Пропуски

print('Информация о пропусках:')

display(data.isnull().sum())

print(30*' =')

#Полные дубликаты 

print(f'Полных дубликатов: {data.duplicated().sum()}')
                            
print(30*' =')


# - Приводим название столбцов к нижнему регистру, также преобразуем названия столбцов :

# In[4]:


data = data.rename(columns = {'RowNumber': 'row_number', 'CustomerId':'custom_id', 'Surname':'surname', 'CreditScore':'credit_score', 'NumOfProducts':'num_of_products', 'HasCrCard':'has_cr_card', 'IsActiveMember':'is_active_member',
                            'EstimatedSalary':'estimated_salary' })

data.columns = map(str.lower, data.columns)

# заполняем пропуски нулевым значение в значениях Tenure (сколько лет человек является клиентом банка)
data['tenure'] = data['tenure'].fillna(0)

display(data.head())
display(data.info())


# Данные кооректны, пропусков и явных дубликатов нет

# ### Подготовка данных для машинного обучения 

# - Устианавливаем целевой признак - 'exited' - факт ухода Клиента из Банка

# - Удаляем столбцы данные которых не понадобятся в исследовании

# In[5]:


data_new = data.drop(['custom_id','surname', 'row_number'], axis=1) 


display(data_new.head())


# - Преобразуем категориальные признаки в количественные для gender, geography и age используя метод OHE:  
# 

# In[6]:


data_new = pd.get_dummies(data_new, drop_first=True)

display(data_new.head())

data_new.shape


# ### Создаем учебную, валидационную и тестовые выборки 

# In[7]:


# разобьем данные на признаки и целевой признак  
features = data_new.drop(['exited'], axis=1)
target = data_new['exited']

display(features.head())
display(target.head())


# In[8]:


# валидационная выборка (в пропорции 60:40 от общих данных)
features_train, features_valid_test, target_train, target_valid_test,  = train_test_split(features, target, 
                                                                          train_size=0.60, random_state=12345, stratify=target) 
print('Признаки обучающей выборки:', features_train.shape)
print('Целевой признак обучающей выборки:', target_train.shape)
print('Признаки валидационной выборки:', features_valid_test.shape)
print('Целевой признаки валидационной выборки:', target_valid_test.shape)


# In[9]:


# тестовая выборка (в пропорции 50:50 от валидационной)
features_valid, features_test, target_valid, target_test = train_test_split(features_valid_test,
                                                    target_valid_test,
                                                    train_size=0.5,
                                                    random_state=12345, stratify=target_valid_test) 
                                                                      

print('Признаки и целевые признаки по валидационной и целевой выборкам:', features_valid.shape, target_valid.shape, features_test.shape, target_test.shape)


# ### Масштабирование 

# - Поскольку у формат колличественных данных значительно отличается друг от друга необходимо применить мастшатбирование, чтобы машина не сделела ошибочных предположении о важности данных чей удельный вес выше. 

# In[10]:


# выделим количественные признаки 
numeric = ['credit_score', 'age', 'tenure', 'balance', 'num_of_products', 'estimated_salary']


# In[11]:


# масштабируем на обучающей выборке
scaler = StandardScaler()
scaler.fit(features_train[numeric])


# In[12]:


#Масштабируем количественные признаки обучающей выборки 
features_train[numeric] = scaler.transform(features_train[numeric])
features_train.head()


# In[13]:


#Масштабируем количественные признаки валидационной выборки 
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_valid.head() 


# In[14]:


#Масштабируем количественные признаки тестовой выборки 
features_test[numeric] = scaler.transform(features_test[numeric])
features_test.head()


# Масштабированные данные для обучающей, валидационной и тестовой выборок получены

# ## Исследование задачи

# ### Обучаем модели на обучающей выборке и находим точность каждой из них на валидационной  

# In[15]:


model_tree = DecisionTreeClassifier(random_state=123)
tree_score = model_tree.fit(features_train, target_train).score(features_valid, target_valid)
    
model_forest = RandomForestClassifier(random_state=12345, n_estimators = 100)
forest_score = model_forest.fit(features_train, target_train).score(features_valid, target_valid)
    
model_regression = LogisticRegression(solver = 'liblinear')
regression_score = model_regression.fit(features_train, target_train).score(features_valid, target_valid)
    
    
print("Точность:" "Дерево решений", tree_score, "Случайный лес ", forest_score, "Логистческая регрессия", regression_score)


# - проверяем балланс классов 

# In[16]:


target_train.value_counts(normalize = 1) 


# In[17]:


target_valid.value_counts(normalize = 1)


# Вывод: у классов наблюдается дисбалланс, поэтому очевидно что предсказания моделей будет склоняться в пользу варианта ответа - 0 

# ### Проверка на адекватность моделей

# In[18]:


# модель решающее дерево
model_tree = DecisionTreeClassifier(random_state=1234)
model_tree.fit(features_train, target_train)
model_tree_class_frequency = pd.Series(model_tree.predict(features_valid)).value_counts(normalize = 1)

print(model_tree_class_frequency)


# In[19]:


# модель случайный лес 
model_forest = RandomForestClassifier(random_state=12345, n_estimators = 100)
model_forest.fit(features_train, target_train)
model_forest_class_frequency = pd.Series(model_forest.predict(features_valid)).value_counts(normalize = 1)
print(model_forest_class_frequency)


# In[20]:


# модель логистической регрессии 
model_regression = LogisticRegression(solver = 'liblinear')
model_regression.fit(features_train, target_train)
model_regression_class_frequency = pd.Series(model_regression.predict(features_valid)).value_counts(normalize = 1)
print(model_regression_class_frequency)


# Самая высокая точность у логистической регресии 

# - Сравним качество предсказаний моделей с точностью константной модели

# In[21]:


target_predict_constant = pd.Series([0]*len(target_valid))
accuracy_score_constant = accuracy_score(target_valid, target_predict_constant)

print(accuracy_score_constant)


# Точность константной модели является првктичеки такой же как и точность других моделей, что указывает на неадекаватность и на возможный дисбаланс классов в данных моделях

# In[22]:


# создаем матрицу ошибок для дерева решений изучаем полноту, точность и F1 меру 

model_tree = DecisionTreeClassifier(random_state=123)
model_tree.fit(features_train, target_train)
model_tree_prediction = model_tree.predict(features_valid)
confusion_matrix(target_valid, model_tree_prediction)


# In[23]:


print("Полнота" , recall_score(target_valid, model_tree_prediction))
print("Точность", precision_score(target_valid, model_tree_prediction))
print("F1-мера", f1_score(target_valid, model_tree_prediction))
print("AUC-ROC", roc_auc_score(target_valid, model_tree_prediction))


# In[24]:


# создаем матрицу ошибок для случайного леса изучаем полноту, точность и F1 меру 

model_forest = RandomForestClassifier(random_state=1234, n_estimators = 100)
model_forest.fit(features_train, target_train)
model_forest_prediction = model_forest.predict(features_valid)
confusion_matrix(target_valid, model_forest_prediction)


# In[25]:


print("Полнота" , recall_score(target_valid, model_forest_prediction))
print("Точность", precision_score(target_valid, model_forest_prediction))
print("F1-мера", f1_score(target_valid, model_forest_prediction))
print("AUC-ROC", roc_auc_score(target_valid, model_forest_prediction))


# In[26]:


# создаем матрицу ошибок для логистической регрессии 

model_regression  = LogisticRegression(solver = 'liblinear')
model_regression.fit(features_train, target_train)
model_regression_prediction = model_regression.predict(features_valid)
confusion_matrix(target_valid, model_regression_prediction)


# In[27]:


print("Полнота" , recall_score(target_valid, model_regression_prediction))
print("Точность", precision_score(target_valid, model_regression_prediction))
print("F1-мера", f1_score(target_valid, model_regression_prediction))
print("AUC-ROC", roc_auc_score(target_valid, model_regression_prediction))


# Значение агрегирующей метрики F1 выше всего у модели Случайный лес, тем не менее данного значения не достаточно, необходимо поработать над точностью, чтобы увеличить значение общего показателя, определяющего предсказания модели 

# ## Борьба с дисбалансом

# ### Для увеличения точности моделей необходимо прийти к баллансу классов данных, дающих положительный и отрицательные ответы в целевом признаке

# In[28]:


def upsample(features, target, repeat, upsampled_сlass):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    
    if upsampled_сlass == 0:
        features_upsampled = pd.concat([features_zeros]* repeat + [features_ones] )
        target_upsampled = pd.concat([target_zeros]* repeat + [target_ones] )
        features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
        
    elif upsampled_сlass == 1:
        features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
        target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
        features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    else:
        features_upsampled = 0
        target_upsampled = 0  
    return features_upsampled, target_upsampled
    
    
features_train_upsampled, target_train_upsampled = upsample(features_train, target_train, 4, 1)
print(target_train_upsampled.value_counts(normalize = 1))
print(target_train_upsampled.shape)    


# Сбалансированное количество классов получено 

# ### Обучение моделей на сбалансированой выборке 

# In[29]:


# Обучаем модели на  upsample выборке и проверяем точность предсказаний на валидационной
model_tree = DecisionTreeClassifier(random_state=123)
tree_score = model_tree.fit(features_train_upsampled, target_train_upsampled).score(features_valid, target_valid)
    
model_forest = RandomForestClassifier(random_state=123, n_estimators = 100)
forest_score = model_forest.fit(features_train_upsampled, target_train_upsampled).score(features_valid, target_valid)
    
model_regression = LogisticRegression(solver = 'liblinear')
regression_score = model_regression.fit(features_train_upsampled, target_train_upsampled).score(features_valid, target_valid)
    
    
print("Точность:" "Дерево решений", tree_score, "Случайный лес ", forest_score, "Логистческая регрессия", regression_score)


# Точность моделей немного увеличилась. Посмторим изменилось ли качество предсказаний 

# In[30]:


# Обучаем модель решающее дерево на сбалансированной выборке изучаем полноту, точность и F1  
model_tree_upsampled = DecisionTreeClassifier(random_state=123)
model_tree_upsampled.fit(features_train_upsampled, target_train_upsampled)
model_tree_upsampled_prediction = model_tree_upsampled.predict(features_valid)

print("Полнота" , recall_score(target_valid, model_tree_upsampled_prediction))
print("Точность", precision_score(target_valid, model_tree_upsampled_prediction))
print("F1-мера", f1_score(target_valid, model_tree_upsampled_prediction))
print("AUC-ROC", roc_auc_score(target_valid, model_tree_upsampled_prediction))


# In[31]:


# Обучаем модель случайный лес на сбалансированной выборке изучаем полноту, точность и F1  

model_forest_upsampled = RandomForestClassifier(random_state=1234, n_estimators = 100)
model_forest_upsampled.fit(features_train_upsampled, target_train_upsampled)
model_forest_upsampled_prediction = model_forest_upsampled.predict(features_valid)

print("Полнота" , recall_score(target_valid, model_forest_upsampled_prediction))
print("Точность", precision_score(target_valid, model_forest_upsampled_prediction))
print("F1-мера", f1_score(target_valid, model_forest_upsampled_prediction))
print("AUC-ROC", roc_auc_score(target_valid, model_forest_upsampled_prediction))


# In[32]:


# Обучаем модель Логистическая регрессия на сбалансированной выборке изучаем полноту, точность и F1  

model_regression_upsampled  = LogisticRegression(solver = 'liblinear')
model_regression_upsampled.fit(features_train_upsampled, target_train_upsampled)
model_regression_upsampled_prediction = model_regression_upsampled.predict(features_valid)

print("Полнота" , recall_score(target_valid, model_regression_upsampled_prediction))
print("Точность", precision_score(target_valid, model_regression_upsampled_prediction))
print("F1-мера", f1_score(target_valid, model_regression_upsampled_prediction))
print("AUC-ROC", roc_auc_score(target_valid, model_regression_upsampled_prediction))


# Показатели моделей стали лучше. Самое высокое значение метрики F1 - 0.62 (а также других метрик) у модели Случайный лес, что выше целевого значение на которое мы ориентируемся при улучшении моделей. Также значение AUC-ROC у модели случанйый лес выше чем у остальных моделей, что указывает на то что данная модель яялется оптимальной для решения поставленной задачи

# In[1]:


#используем инструмент GridSearchCv для поиска лучших параметров модели "Случайный лес" 
X_train = features_train_upsampled
y_train = target_train_upsampled

clf = RandomForestClassifier() 
parametrs = { 'n_estimators': range (0, 10, 1),
              #'max_depth': range (1,13, 2),
              #'min_samples_leaf': range (1,8),
              #'min_samples_split': range (2,10,2) }

grid = GridSearchCV(clf, parametrs, cv=5)
grid.fit(X_train, y_train)

grid.best_params_


# - Обучим финальную модель 

# In[34]:


model_final = RandomForestClassifier(
     class_weight = 'balanced', max_depth= 9,  n_estimators = 9, random_state=1234)
model_final.fit(features_train_upsampled, target_train_upsampled)

model_final_predict = model_final.predict(features_valid)

print("Полнота" , recall_score(target_valid, model_final_predict))
print("Точность", precision_score(target_valid, model_final_predict))
print("F1-мера", f1_score(target_valid, model_final_predict))
print("AUC-ROC", roc_auc_score(target_valid, model_final_predict))


# - Проверяем финальную модель на адекватность 

# In[35]:


target_predict_const = pd.Series([0]*len(target_valid))
target_predict_const.value_counts() 


# In[36]:


#Сравним показатель точности (accuracy_score) константной модели и финальной
print('accuracy_score константой модели:', accuracy_score(target_valid, target_predict_const))
print('accuracy_score финальной модели:', accuracy_score(target_valid, model_final_predict))

#Сравним AUC-ROC  - константная модель содержит только негативные ответы, поэтому важно сравнить показатель с финальной моделью
print('AUC-ROC константой модели:', roc_auc_score(target_valid, target_predict_const))
print('AUC-ROC финальной модели:', roc_auc_score(target_valid, model_final_predict))


# Финальная модель показывает результаты лучше, чем константная модель — модель можно считать адекватной.
# 
# 

# Общий вывод: В полученных данных были обнаружены существенные расхождения в количественных показателях, поэтому было применено масштабирование. В данных наблюдался значительный дисбаланс классов по целевому признаку в примерном отношении 20:80, поэтому обученные модели не проходили проверку на адекватность. Мы устранили дисбаланс методом  upsampling, увеличив количество позитивного класса в 4 раза. После обучения на новых данных все модели показали лучшие показатели точности, полноты, F1, параметр ROC-AUR так же стал выше, что указывает на рост True Positive вариантов ответов. 
# Абсолютно лучшей оказалась модель - Случайный лес со следующими метриками: 
# 
# Полнота 0.5735294117647058
# Точность 0.6763005780346821
# F1-мера 0.6206896551724138
# AUC-ROC 0.751588826485368
# 
# Далее используя инструмент GridSearchCv были подобраны оптимальные набор параметров для данной модели: 
# 
# Максимальная глубина дерева - max_depth': 11,
# Минимальное количество листьев -  'min_samples_leaf': 1,
# Минимальное число образцов для сплита 'min_samples_split': 2,
# Число деревьев в лесу -  'n_estimators': 9
# 
# Финальная модель прошла проверку на адекватность в сравнении с константной моделью:
# 
# accuracy_score константой модели: 0.796
# accuracy_score финальной модели: 0.8285
# AUC-ROC константой модели: 0.5
# AUC-ROC финальной модели: 0.7692321903635826
# 
# 

# ## Тестирование модели

# In[37]:


# Тестируем финальную модель на тестовой выборке 
model_final_predict = model_final.predict(features_test)

print("Полнота" , recall_score(target_test, model_final_predict))
print("Точность", precision_score(target_test, model_final_predict))
print("F1-мера", f1_score(target_test, model_final_predict))
print("AUC-ROC", roc_auc_score(target_test, model_final_predict))


# Финальная модель прошла тестирование, достигла заданой метрики F1>0.59 и по другим показателям также является адекватной. Метрика  AUC-ROC увеличивалась в ходе улучшения модели, что указывает на рост правильных предсказаний.
# Модель характеризуется высоким показателем полноты = 0.67 (min = 0, max = 1), поэтому она с высокой вероятностью предскажит уход клиента из банка.
# Показатель точности не высокий = 0.527 (min = 0, max = 1) — модель верно предсказывает только половину ухода клиентов.
# Тем не менее, полученная модель поможет лучше определять килентов, которые могут уйти из Банка в ближайшее время. 
# 

# In[ ]:




