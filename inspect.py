#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Lingli Duan-HW1-Programming
from __future__ import division
import sys
if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2] 
#     print("The input file is: %s" % (infile)) 
#     print("The output file is: %s" % (outfile))


# In[62]:


import csv
label_counts={}

with open(infile) as csvfile:
    csvReader= csv.DictReader(csvfile)
    label_name = csvReader.fieldnames[-1]
    for row in csvReader:       
        if row[label_name] not in label_counts:
            label_counts[row[label_name]] = label_counts.get(label_name,0)+1
        else:
            label_counts[row[label_name]]+=1
       


# In[63]:


label_counts


# In[64]:


import math
entropy = 0
number_of_examples = 0
for count in label_counts.values():
    number_of_examples +=count
number_of_examples
for label, count in label_counts.items():
    if count >0 and number_of_examples>0:
        entropy -= (count/number_of_examples) * math.log(count/number_of_examples,2)
entropy = round(entropy,12)


# In[65]:


error=1-(max(label_counts.values())/number_of_examples)
error=round(error,12)


# In[66]:


entropy


# In[67]:


error


# In[48]:


with open(outfile, mode='w') as f_output:
    f_output.write("entropy: "+str(entropy)+"\n")
    f_output.write("error: "+str(error)+"\n")


# In[ ]:




