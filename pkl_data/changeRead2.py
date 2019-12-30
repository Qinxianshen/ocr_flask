#-*- coding=utf-8 -*-

#将原本在python3 下的pickle转化成python2版本的
import pickle
with open('./address1.pkl', 'rb') as f:    
    w = pickle.load(f)
    pickle.dump(w, open('./address2.pkl',"wb"), protocol=2)

with open('./others1.pkl', 'rb') as f:    
    w = pickle.load(f)
    pickle.dump(w, open('./others2.pkl',"wb"), protocol=2)
