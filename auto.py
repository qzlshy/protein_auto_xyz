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

dis_t=dis[2:]
angle_t=angle[1:]
dhd_t=dhd

t1=dis_t*np.cos(angle_t)
t2=dis_t*np.cos(dhd_t)*np.sin(angle_t)
t3=dis_t*np.sin(dhd_t)*np.sin(angle_t)

tt=[t1,t2,t3]
tt=np.array(tt)
tt=tt.T

c=[]
for i in range(3):
    c.append(xyz[i])

for i in range(len(tt)):
    k=i+3
    mk=c[k-1]-c[k-2]
    mk_1=c[k-2]-c[k-3]
    mk_n=mk/np.sqrt(np.sum(mk*mk))
    nk=np.cross(mk_1,mk_n)
    nk_n=nk/np.sqrt(np.sum(nk*nk))
    nk_mk_n=np.cross(nk_n,mk_n)
    R=[mk_n,nk_mk_n,nk_n]
    R=np.array(R)
    t=np.dot(R.T,tt[i].T)+c[k-1]
    c.append(t)


