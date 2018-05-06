This project was done as a course project for CSE 592 Convex Optimization , Spring 2018 at Stony Brook University computer science.
This repository contains the code for a saddle point optimizer proposed by Nestorov. We have applied this optimizer to train Wasserstein GAN. For details please refer to main.pdf in "report" folder

# Applying Dual-Averaging Sub-gradient method to

# saddle point problem of Generative adversarial

# networks

```
Amol Damare
Department of Computer Science
Stony Brook University
SBU Id:
adamare@cs.stonybrook.edu
```
```
Arjun Krishna
Department of Computer Science
Stony Brook University
SBU Id:
arjkrishna@cs.stonybrook.edu
```
## Abstract

```
Training of Generative Adversarial Networks is hard. It is one of the open research
problem in the field of deep learning. The objective of GAN is modeled as saddle
point problem. In this paper we have applied already existing algorithms for saddle
point problem from optimization theory to GAN. Specifically we have implemented
SDA method proposed by 
Algorithm 2 Algorithm for Dual-Averaging for Saddle point problem

```
1:S 0 = 0
2:?? 0 ,?? 1 = 1
3:Choose? > 0
4:For k=0..T
5:Compute
gk= (guk,?gvk)
6:Compute
Sk+1=Sk+gk
7:Compute
```
```
?k?+1=
```
```
?k
```
```
i=
```
### 1

```
??i
8:Compute
?k+1=??k?+
9:Compute
xk+1=??k+1(?Sk+1)
```
10: End For
11: returnxT

````
## 5 Conclusion

We were able to successfully apply an existing optimization algorithm for Saddle point problem
to Wasserstein GAN with relatively simple model( 1 or 2 layer deep). But we were not able to
replicate same for cross-entropy GAN objective or a more complex GAN model like DC-GAN. When
compared with existing optimization techniques used to solve GAN we found that they perform very
similarly.


