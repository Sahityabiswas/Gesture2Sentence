
import pickle
with open('vid_splits_FDMSE-ISL.pkl','rb') as f: s=pickle.load(f)
with open('vid_class_FDMSE-ISL.pkl','rb') as f: c=pickle.load(f)
print('Train vids:', len(s['train']), '| Test vids:', len(s['test']))
print('Total classes:', len(set(c.values())))
print('Samples per class avg:', round(len(s['train'])/len(set(c.values())),1))
