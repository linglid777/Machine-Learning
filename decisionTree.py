#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ml601- hw2 - linglid
from __future__ import division, print_function
import csv
import numpy as np

def open_csv(file_path):
    
    with open(file_path) as input_file:
        csvReader = csv.reader(input_file)
        data = np.array(list(csvReader))
        return data


# In[2]:


def entropy(feature):
    from collections import Counter
    value_count=Counter()
    for v in feature:
        value_count[v]+=1
#     print(value_count)
    entro = 0
    length = len(feature)
    for count in value_count.values():
        ratio = count/length
#         if(ratio !=0.0 & ratio!=1.0):
        entro += -ratio* np.log2(ratio)
    
    return entro


# In[3]:


def mutual_info(featureY, featureX):
    return entropy(featureY)+entropy(featureX)-entropy(list(zip(featureY, featureX)))


# In[4]:


class node():
    def __init__(self, depth, dataset,feature, value):
        from collections import Counter
        self.left = None
        self.right = None
        self.current_depth = depth 
        self.data = dataset
        self.feature = feature
        self.value = value
        
        if self.feature!=None:
            self.condition = str(self.feature)+"="+str(value)
        else:
            self.condition=None
       
        self.value_count = Counter()
        for v in self.data[1:,-1]:
            self.value_count[v]+=1
        self.prediction = max(self.value_count.items(),key = lambda x:x[1])[0]


# In[5]:



def build_stump(root, current_depth):
    
#     if len(root.data[0])-1 < max_depth:# #of columns fewer than the required max_length
#         max_depth = len(data[0])-1---put in buildTree()
    
    if max_depth==0:
        return root
    if len(root.data[0])>1 and current_depth < max_depth:
        
        is_single_label = True if len(np.unique(root.data[1:,-1]))==1 else False #sign of if the current data is single-labelled 
    
        if  is_single_label==False:
            feature_MI = {}
            for i in range(len(root.data[0])-1):
                feature_MI[i] = mutual_info(root.data[1:,-1],root.data[1:,i])# a dict of mutual info combinations
#                 print(str(i)+":"+str(feature_MI[i]))
            remove_col_index = max(feature_MI.items(), key = lambda x:x[1])[0]
#             print("remove_col_index: "+ str(remove_col_index))
            unique_value = np.unique(root.data[1:,remove_col_index])
#             for i in unique_value:
#                 print(i)
            if len(unique_value)==2:
        
                current_depth+=1

                left_data=[row for row in root.data if row[remove_col_index]==unique_value[0] or row[remove_col_index]==root.data[0][remove_col_index]]
                left_data = np.delete(left_data,obj=remove_col_index,axis=1)
    #             left_logic_expression = root.data[0][remove_col_index] + " = " + unique_value[0]
                root.left = node(current_depth, left_data, str(root.data[0][remove_col_index]), unique_value[0])
    #             print(left_data)
    #             print(root.left.current_depth)

                right_data=[row for row in root.data if row[remove_col_index]==unique_value[1] or row[remove_col_index]==root.data[0][remove_col_index]]
                right_data = np.delete(right_data,obj=remove_col_index,axis=1)
    #             right_logic_expression = root.data[0][remove_col_index] + " = " + unique_value[1]
                root.right = node(current_depth, right_data,str(root.data[0][remove_col_index]), unique_value[1])   

                build_stump(root.left, current_depth)
                build_stump(root.right, current_depth)
                return root
            else:
                return


# In[8]:


def predict(root, data):
    
    def predict_row(root,row_dict):
        if root.left==None and root.right == None:
            row_dict['label']=root.prediction
            return
        if row_dict[root.left.feature] == root.left.value:
            predict_row(root.left, row_dict)
        elif row_dict[root.right.feature] == root.right.value:
            predict_row(root.right, row_dict)
    
    list_of_predicted_labels = []
    for row in data[1:,:]:
        row_dict = dict(zip(data[0],row))
        predict_row(root, row_dict)
        list_of_predicted_labels.append(row_dict['label'])
    return list_of_predicted_labels


# In[9]:


def metrics(predicted_labels, true_labels):
    both_labels = list(zip(predicted_labels,true_labels))
    error_count = 0
    for labels in both_labels:
        if labels[0] != labels[1]:
            error_count+=1
    return error_count/len(both_labels)


# In[26]:



import sys
if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_labels=sys.argv[4]
    test_labels=sys.argv[5]
    metrics_file =sys.argv[6]
    

#load training data
    train_data = open_csv(train_file)

#create root node    
    root = node(0,train_data,None,None)
    if len(root.data[0])-1 < max_depth:# #of columns fewer than the required max_length
        max_depth = len(root.data[0])-1
        
#build tree
    trained_root = build_stump(root,root.current_depth)
    train_predicted_labels = predict(trained_root, train_data)
    with open(train_labels,"w") as train_label_output:
        for label in train_predicted_labels:
            train_label_output.write(label+'\n')
            
#predict test_data
    test_data = open_csv(test_file)
    test_predicted_labels = predict(trained_root, test_data)
    with open(test_labels,"w") as test_label_output:
        for label in test_predicted_labels:
            test_label_output.write(label+'\n')

#write metrics.txt
    with open(metrics_file,"w")as metrics_output:
        metrics_output.write("error(train): "+str(metrics(train_predicted_labels, train_data[1:,-1]))+'\n')
        metrics_output.write("error(test): "+str(metrics(test_predicted_labels, test_data[1:,-1]))+'\n')

    


# In[27]:


# define print_description()
    des_dict = {value:count for (value, count) in trained_root.value_count.items() } # structure the data description format based on rootNode 
    des_dict

    def print_description(node):

        for value in des_dict.keys():
            if value not in node.value_count.keys():
                des_dict[value]=0
            else:
                des_dict[value] = node.value_count[value]
        subHeading = subHeading = [str(count)+' '+value for value,count in des_dict.items()]
        description=('['+str(subHeading[0])+'/'+str(subHeading[-1])+']')
        print(description)
   
    def print_stump(tree):
        for i in range(tree.current_depth):
            print("|", end = " ")
        if tree.condition !=None:
            print(tree.condition+":", end=" ")
        print_description(tree)
        
        if tree.left != None:
            print_stump(tree.left)
        if tree.right != None:
            print_stump(tree.right)
    
    print_stump(trained_root)

