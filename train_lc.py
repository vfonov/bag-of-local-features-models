#! /bin/bash
import os
import glob

from bagnets.pytorch import bagnet17

import torch
import torch.optim as optim
import torch.nn as nn

from skimage.io import imread
import numpy as np

model = bagnet17(True).cuda()
patchsize=17
batch_size=2000 # 

total_epochs=100

num_classes = 10 # max number of classes

# train a simple linear classifier, using CPU
# iterate over images

# simplest linear classifier
simple_cls = nn.Linear(1000, num_classes)
simple_cls = simple_cls.cuda()
simple_cls.train()
loss =  nn.BCEWithLogitsLoss()  #  nn.CrossEntropyLoss()
#optimizer = optim.SGD(simple_cls.parameters(),lr=0.001)

optimizer = optim.Adam(simple_cls.parameters(),lr=0.001)

for epoch in range(total_epochs):
    avg_loss = 0.0
    avg_loss_cnt = 0
    avg_acc = 0
    for i in  glob.glob('img/object_*_*.jpg'):
        img_avg_loss = 0.0
        img_avg_cnt = 0
        _,c,_ = os.path.basename(i).split('_')
        c=int(c) if c!='bg' else 0
        
        img = imread(i, as_gray=False)
        img = img / 255.
        img -= np.array([0.485, 0.456, 0.406])[None, None, :]
        img /= np.array([0.229, 0.224, 0.225])[None, None, :]
        x, y, _  = img.shape
        padded_image = np.zeros((x + patchsize - 1, y + patchsize - 1, 3))
        padded_image[(patchsize-1)//2:(patchsize-1)//2 + x, (patchsize-1)//2:(patchsize-1)//2 + y,:] = img
        image = padded_image[None].astype(np.float32)
        inp = torch.from_numpy(image).cuda()

        #patches = inp.permute(0, 2, 3, 1)
        patches = inp
        patches = patches.unfold(1, patchsize, patchsize).unfold(2, patchsize, patchsize)
        num_rows = patches.shape[1]
        num_cols = patches.shape[2]
        patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

        # could be optimized
        target = torch.zeros(len(patches), num_classes)
        target.scatter_(1, torch.full((len(patches),1) , c, dtype=torch.long),1)
        target = target.cuda()
        # compute logits for each patch

        for batch_patches,batch_target in zip(torch.split(patches, batch_size),torch.split(target, batch_size)):
            optimizer.zero_grad()
            with torch.no_grad():
                out = model(batch_patches)
            out_lin = simple_cls(out)

            out_loss = loss(out_lin, batch_target)

            avg_loss+=float(out_loss)
            avg_loss_cnt+=1
            img_avg_loss+=float(out_loss)
            img_avg_cnt+=1
            
            out_c=out_lin.argmax(1)
            avg_acc+=float((out_c==c).sum())/len(out_c)

            out_loss.backward()
            optimizer.step()
        #img_avg_loss/=img_avg_cnt
        #print(c,img_avg_loss)
    avg_loss/=avg_loss_cnt
    avg_acc/=avg_loss_cnt
    print("{}/{}: {} acc:{}".format(epoch,total_epochs,avg_loss,avg_acc))
# TODO: save the model
simple_cls=simple_cls.cpu()
