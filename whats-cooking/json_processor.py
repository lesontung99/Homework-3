import json
import numpy as np
import time
from itertools import chain


start = time.time()
with open("Questions\CIS419-master\Assignment3\whats-cooking\\test.json",'r') as json_File :
    sample=json.load(json_File)

id_test = [d['id'] for d in sample if 'id' in d]
ing_test = [d['ingredients'] for d in sample if 'ingredients' in d]

with open("Questions\CIS419-master\Assignment3\whats-cooking\\train.json",'r') as json_File :
    train_sample=json.load(json_File)
id = [d['id'] for d in train_sample if 'id' in d]
ing = [d['ingredients'] for d in train_sample if 'ingredients' in d]
dish = [d['cuisine'] for d in train_sample if 'cuisine' in d]

#print (ing)
flatten = list(chain.from_iterable(ing))
atto = np.unique(flatten)
#print (len(atto))
id = np.array(id)
id_test = np.array(id_test)

id = id.reshape((len(id),1))
id_test = id_test.reshape((len(id_test),1))
#(6k variable, 33k category...)
dish_list = np.unique(dish)
dish_list = dish_list.reshape(len(dish_list),1)
dish_index = np.array(range(len(dish_list))).reshape(len(dish_list),1)
translate = np.concatenate((dish_index,dish_list), axis = 1)
np.savetxt("Questions/CIS419-master/Assignment3/whats-cooking/dish_translate.dat", translate, fmt='%s')

#print (len(dish_list))
'''dish_num = np.empty((len(dish),1))
for cat in range (len(dish_list)):
    for i in range (len(dish)):
        if dish_list[cat] == dish[i]:
            dish_num[i,0] = cat
ing_arr = np.empty((len(ing),len(atto)))
for cat in range(len(atto)):
    for i in range (len(ing)):
        if atto[cat] in ing[i]:
            ing_arr[i,cat] = 1
        else:
            ing_arr[i,cat] = 0
np.savetxt("Questions/CIS419-master/Assignment3/whats-cooking/dish_train.dat", np.concatenate((id,ing_arr,dish_num), axis = 1), fmt = '%d')
'''
'''ing_test_arr = np.empty((len(ing_test),len(atto)))
for cat in range(len(atto)):
    for i in range (len(ing_test)):
        if atto[cat] in ing_test[i]:
            ing_test_arr[i,cat] = 1
        else:
            ing_test_arr[i,cat] = 0
np.savetxt("Questions/CIS419-master/Assignment3/whats-cooking/dish_test.dat", np.concatenate((id_test, ing_test_arr), axis = 1), fmt = '%d')'''
finish = time.time() - start
print(finish)
