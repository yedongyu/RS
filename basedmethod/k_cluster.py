# coding: utf-8

# In[33]:


import numpy as np
import csv
#time=np.array([891804959,891027359,888348959,887052959,885584159,884028959,882041759,875649599,876945599,878241599,879623999,880919999,882215999,883511999,884894399,886190399,887572799,888695999,889991999,891287999])

#User=np.zeros((943,20))
#User1=np.zeros((943,20))
#j=0
#U=0
# k=0
# Q = open('sort_data.txt','w')
# with open('u.data','r') as s:
#     data = s.readlines()
#     for line in sorted(data, key = lambda line: (int(str(line.split('\t')[0])), int(str(line.split('\t')[3])))):
#         Q.write(line)
# #         Q.write("\n")
# Q.close()



# # In[4]:
# U=0
# k=0
# count_user=0
# count=0
# with open('sort_data.txt', 'r') as g:
#     data1 =g.readlines()
#     D1=data1[0].split('\t')
#     id_former=int(D1[0])
#     count_user_former=0
#     train_file = open('train_data.tsv', 'w')
#     test_file = open('test_data.tsv', 'w')
#     while(U<943):
#         D=data1[k].split('\t')
#         if len(D)!=4:
#             continue
#         current_id=int(D[0])
#         if(current_id!=id_former):
#            # print(current_id)
#             count_user_train =count_user_former+ int(count * 0.9)
#             m = -1
#             w = -1
#             for j in range(count_user_former, count_user_train):
#                 Dp=data1[j].split('\t')
#                 if(int(Dp[2])>=3):
#                     m = 1
#                 if(int(Dp[2])<3):
#                     m = 0
#                 train_file.write("%s %s %s\n" %(Dp[0], Dp[1], m))
#             for p in range(count_user_train,count_user):
#                 Dp=data1[j].split('\t')
#                 if(int(Dp[2])>=3):
#                     w = 1
#                 if(int(Dp[2])<3):
#                     w = 0
#                 test_file.write("%s %s %s\n" %(Dp[0], Dp[1], w))
#             id_former=current_id
#             U=U+1 
#             count_user_former = count_user
#             count=0
#         k+=1
#         count_user+=1
#         count+=1
#         if(k==len(data1)):
#             break
# train_file.close()
# test_file.close()



U=0
k=0
train_1 = open('./data/k8/traink8_1.tsv', 'w')
train_2 = open('./data/k8/traink8_2.tsv', 'w')
train_3 = open('./data/k8/traink8_3.tsv', 'w')
train_4 = open('./data/k8/traink8_4.tsv', 'w')
train_5 = open('./data/k8/traink8_5.tsv', 'w')
train_6 = open('./data/k8/traink8_6.tsv', 'w')
train_7 = open('./data/k8/traink8_7.tsv', 'w')
train_8 = open('./data/k8/traink8_8.tsv', 'w')


label = open('./data/k8/k8_label.txt', 'r').readlines()


with open('./data/train_data.tsv', 'r') as g:
    data1 =g.readlines()
    D1=data1[0].split(' ')
    id_former=int(D1[0])
    while(U<111630):
        D=data1[k].split(' ')
        if len(D)!=3:
            continue
        current_id=int(D[0])
        if(current_id!=id_former):
            id_former=current_id
            U=U+1
        l = int(label[U])
        if l == 0:
            train_1.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 1:
            train_2.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 2:
            train_3.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 3:
            train_4.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 4:
            train_5.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 5:
            train_6.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 6:
            train_7.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 7:
            train_8.write("%s %s %s" %(D[0], D[1], D[2]))
        k+=1
        if(k==len(data1)):
            break

train_1.close()
train_2.close()
train_3.close()
train_4.close()
train_5.close()
train_6.close()
train_7.close()
train_8.close()


U=0
k=0

test_1 = open('./data/k8/testk8_1.tsv', 'w')
test_2 = open('./data/k8/testk8_2.tsv', 'w')
test_3 = open('./data/k8/testk8_3.tsv', 'w')
test_4 = open('./data/k8/testk8_4.tsv', 'w')
test_5 = open('./data/k8/testk8_5.tsv', 'w')
test_6 = open('./data/k8/testk8_6.tsv', 'w')
test_7 = open('./data/k8/testk8_7.tsv', 'w')
test_8 = open('./data/k8/testk8_8.tsv', 'w')

label = open('./data/k8/k8_label.txt', 'r').readlines()



with open('./data/test_data.tsv', 'r') as g:
    data1 =g.readlines()
    D1=data1[0].split(' ')
    id_former=int(D1[0])
    while(U<111630):
        D=data1[k].split(' ')
        if len(D)!=3:
            continue
        current_id=int(D[0])
        if(current_id!=id_former):
            id_former=current_id
            U=U+1
        l = int(label[U])
        if l == 0:
            test_1.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 1:
            test_2.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 2:
            test_3.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 3:
            test_4.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 4:
            test_5.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 5:
            test_6.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 6:
            test_7.write("%s %s %s" %(D[0], D[1], D[2]))
        elif l == 7:
            test_8.write("%s %s %s" %(D[0], D[1], D[2]))
        k+=1
        if(k==len(data1)):
            break
test_1.close()
test_2.close()
test_3.close()
test_4.close()
test_5.close()
test_6.close()
test_7.close()
test_8.close()

# label = open('label.txt', 'r').readlines()
# count = np.zeros((1,4))
# for i in range(len(label)):
#     if int(label[i]) == 0:
#         count[0][0]+=1
#     if int(label[i]) == 1:
#         count[0][1]+=1
#     if int(label[i]) == 2:
#         count[0][2]+=1
#     if int(label[i]) == 3:
#         count[0][3]+=1
# print(count)



