from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import Bio
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio import SeqIO
import numpy as np

def cal_dis(a,b):
    t=(a-b)**2
    t=np.sqrt(np.sum(t,axis=1))
    return t

def cal_angle(a,b,c):
    x=b-a
    y=c-b
    Lx=np.sqrt(np.sum(x*x,axis=1))
    Ly=np.sqrt(np.sum(y*y,axis=1))
    cos_angle=np.sum(x*y,axis=1)/(Lx*Ly)
    angle=np.arccos(cos_angle)
    return angle

def cal_dhd(a,b,c,d):
    L=np.cross(b-a,c-b)
    R=np.cross(d-c,b-c)
    Lnorm=np.sqrt(np.sum(L*L,axis=1))
    Rnorm=np.sqrt(np.sum(R*R,axis=1))
    S=np.cross(L,R)
    angle_cos=np.sum(L*R,axis=1)/(Lnorm*Rnorm)
    angle=np.arccos(angle_cos)
    t=np.sum(S*(c-b),axis=1)
    cs=t<0
    angle[cs]=-angle[cs]
    return angle

p = PDBParser(PERMISSIVE=0)
pdb_name='cdk4_f.pdb'
s = p.get_structure("1",pdb_name)
s = s[0]['U']
res_list = PDB.Selection.unfold_entities(s, 'R')
aa_list = []
for a in res_list:
    if PDB.is_aa(a):
        aa_list.append(a)

xyz=[]
for a in aa_list:
    N=a['N'].get_vector().get_array()
    CA=a['CA'].get_vector().get_array()
    C=a['C'].get_vector().get_array()
    xyz.append(N)
    xyz.append(CA)
    xyz.append(C)

xyz=np.array(xyz)
dis_id=[]
for i in range(len(xyz)-1):
    dis_id.append([i,i+1])

angle_id=[]
for i in range(len(xyz)-2):
    angle_id.append([i,i+1,i+2])

dhd_id=[]
for i in range(len(xyz)-3):
    dhd_id.append([i,i+1,i+2,i+3])

dis_data=xyz[dis_id]
angle_data=xyz[angle_id]
dhd_data=xyz[dhd_id]

dis=cal_dis(dis_data[:,0],dis_data[:,1])
angle=cal_angle(angle_data[:,0],angle_data[:,1],angle_data[:,2])
dhd=cal_dhd(dhd_data[:,0],dhd_data[:,1],dhd_data[:,2],dhd_data[:,3])

import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self,dis, angle, dhd):
        super(Net, self).__init__()

        self.dis_tensor=torch.from_numpy(dis[2:])
        self.angle_tensor=torch.from_numpy(angle[1:])

        self.dhd_vt=[]
        for i in range(len(dhd)):
            t=torch.from_numpy(np.array(dhd[i])).view(1)
            t=Variable(t, requires_grad=True)
            self.dhd_vt.append(t)

        for i in range(int(len(self.dhd_vt)/3)):
            self.dhd_vt[i*3+1].requires_grad = False

        self.dis_v=Variable(self.dis_tensor, requires_grad=False)
        self.angle_v=Variable(self.angle_tensor, requires_grad=False)


    def forward(self, xyz):
        dhd_v=torch.cat(self.dhd_vt,0)
        t1=self.dis_v*torch.cos(self.angle_v)
        t2=self.dis_v*torch.cos(dhd_v)*torch.sin(self.angle_v)
        t3=self.dis_v*torch.sin(dhd_v)*torch.sin(self.angle_v)

        t1=t1.view(1,-1)
        t2=t2.view(1,-1)
        t3=t3.view(1,-1)

        tt=torch.cat((t1,t2,t3),0)
        tt=torch.t(tt)

        c=[]
        for i in range(3):
            c.append(torch.from_numpy(xyz[i]).view(1,3))

        for i in range(len(tt)):
            k=i+3
            mk=c[k-1]-c[k-2]
            mk_1=c[k-2]-c[k-3]
            mk_n=mk/torch.sqrt(torch.sum(mk*mk))
            nk=torch.cross(mk_1,mk_n)
            nk_n=nk/torch.sqrt(torch.sum(nk*nk))
            nk_mk_n=torch.cross(nk_n,mk_n)
            mk_n=mk_n.view(1,-1)
            nk_mk_n=nk_mk_n.view(1,-1)
            nk_n=nk_n.view(1,-1)
            R=torch.cat((mk_n,nk_mk_n,nk_n),0)
            t=torch.matmul(torch.t(R),tt[i].view(3,1)).view(1,3)+c[k-1]
            c.append(t)

        c=torch.cat(c,0)
        return c


net=Net(dis, angle, dhd)



criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.SGD(net.dhd_vt, lr=0.00001, momentum=0.9)
target=100.0
target=np.array(target)
target=torch.from_numpy(target)

running_loss = 0.0
for epoch in range(100):
    optimizer.zero_grad()
    c=net(xyz)
    r=torch.norm(c[0]-c[-1])
    loss = criterion(r, target)
    loss.backward()
    optimizer.step()
    #print(loss.item())

c=c.data.numpy()
for i in range(len(c)):
    print(*c[i])

#for i in range(len(xyz)):
#    print(*xyz[i])
