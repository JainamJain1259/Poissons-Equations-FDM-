import numpy as np
import math as m
import matplotlib.pyplot as plt
import sys

num_mesh = 11     # number of mesh points
pie=m.pi
del_x    = 1.0/(num_mesh-1.0)        # mesh size (Delta_x)
xmesh    = np.zeros(num_mesh)        # Mesh points
u_old    = np.zeros(num_mesh)
sol   = np.zeros(num_mesh)        # solution at the previous time step
u_new_Jacobi   = np.zeros(num_mesh)
u_new_Seidel   = np.zeros(num_mesh)
u_new_SOR   = np.zeros(num_mesh)

SOR1 = np.zeros(num_mesh)
S1 = np.zeros(num_mesh)
J1 = np.zeros(num_mesh)


# Diagonal elements of system matrix
dia   = np.zeros(num_mesh)        # main diagonal elements
upp    = np.zeros(num_mesh)        # upper diagonal
low    = np.zeros(num_mesh)        # lower diagonal

# RHs of the discretized linear algebraic system
rhs    = np.zeros(num_mesh)

# Compute the location of mesh points
for i in range(0, len(xmesh)):
    xmesh[i] = i * del_x

#Exact solution
exact = np.zeros(num_mesh)
for i in range(0,num_mesh-1):
  exact[i] = m.sin(pie*xmesh[i])

iteration_jacobi = 0
iteration_seidel = 0
iteration_SOR = 0

norm_J = 100
norm_S = 100
norm_SOR = 100

low[0] = 0.0
upp[0] = 0.0
dia[0] = 1.0
rhs[0] = 0.0
rhs[num_mesh-1] = 0.0
low[num_mesh-1] = 0.0
upp[num_mesh-1] = 0.0
dia[num_mesh-1] = 1.0
for i in range(1, num_mesh-1):
    dia[i] = 2.0
    low[i] = -1.0
    upp[i] = -1.0
    rhs[i] = (pie*pie*m.sin(pie*xmesh[i]))*del_x*del_x



#==============================THOMAS_ALGORITHM==========================#
dia1    = np.zeros(num_mesh)
rhs1    = np.zeros(num_mesh)
dia1[0] = dia[0]
rhs1[0] = rhs[0]
for i in range(1, num_mesh-1):
        dia1[i] = dia[i] - low[i]*upp[i-1]/dia1[i-1]
        rhs1[i] = rhs[i] - rhs1[i-1]*low[i]/dia1[i-1]
       # print(dia1[i], upp[i],low[i],rhs1[i])
sol[num_mesh-1] = rhs[num_mesh-1]/dia[num_mesh-1]
for i in range(len(dia)-2,-1,-1):
        sol[i] = (rhs1[i]-upp[i]*sol[i+1])/dia1[i]

    # plot the converged results
plt.plot(xmesh,sol,'r-o')
plt.plot(xmesh,exact,'g-o')
plt.xlabel('Mesh Points')
plt.ylabel('Solution')
plt.show()

#==============================JACOBI_ALGORITHM==========================#
u_new_Jacobi = np.zeros(num_mesh)
itererr_J = []
Error_Jacobi = np.zeros(num_mesh)
iter_j = 0
converged = False
while converged == False:
        for i in range(1,num_mesh-1):
            u_new_Jacobi[i] = (rhs[i]-(low[i]*J1[i-1])-(upp[i]*J1[i+1]))/dia[i]

        for i in range(1,num_mesh-1):
            Error_Jacobi[i] = (rhs[i]-low[i]*u_new_Jacobi[i-1]-upp[i]*u_new_Jacobi[i+1]-2*u_new_Jacobi[i])
       
        norm_J = abs(max(Error_Jacobi, key=abs))
        itererr_J.append((iter_j,norm_J))
       
        if norm_J < 0.00000001:                                                        # convergence criteria
          converged = True

        for i in range(1,num_mesh-1):
            J1[i] = u_new_Jacobi[i]
        iter_j = iter_j+1
           # plot the converged results
plt.plot(xmesh,J1,'r-o')
plt.plot(xmesh,exact,'g-o')
plt.xlabel('Mesh Points')
plt.ylabel('Solution')
plt.show()
print('Number of iterations for Jacobi = ')
print(iter_j)

#==============================SEIDEL_ALGORITHM==========================#
u_new_Seidel= np.zeros(num_mesh)
itererr_S =[]
iter_s = 0
Error_Seidel = np.zeros(num_mesh)
converged = False
while converged == False:
        for i in range(1,num_mesh-1):
            u_new_Seidel[i] = (rhs[i]-(low[i]*u_new_Seidel[i-1])-(upp[i]*S1[i+1]))/dia[i]

        for i in range(1,num_mesh-1):
            Error_Seidel[i] = (rhs[i]-low[i]*u_new_Seidel[i-1]-upp[i]*u_new_Seidel[i+1]-2*u_new_Seidel[i])
       
        norm_S = abs(max(Error_Seidel, key=abs))
        itererr_S.append((iter_s,norm_S))
       
        if norm_S < 0.00000001:                                                        # convergence criteria
          converged = True

        for i in range(1,num_mesh-1):
            S1[i] = u_new_Seidel[i]
        iter_s = iter_s+1
                # plot the converged results
plt.plot(xmesh,S1,'r-o')
plt.plot(xmesh,exact,'g-o')
plt.xlabel('Mesh Point')
plt.ylabel('Solution')
plt.show()
print('Number of iterations for Seidel = ')
print(iter_s)


#==============================SOR_ALGORITHM==========================#
u_new_SOR= np.zeros(num_mesh)
w=1.5
itererr_SOR =[]
iter_sor = 0
Error_SOR = np.zeros(num_mesh)
converged = False
while converged == False:
        for i in range(1,num_mesh-1):
            u_new_SOR [i] = (1-w)*SOR1[i]+w*((rhs[i]-(low[i]*u_new_SOR [i-1])-(upp[i]*SOR1[i+1]))/dia[i])

        for i in range(1,num_mesh-1):
            Error_SOR[i] = (rhs[i]-low[i]*u_new_SOR[i-1]-upp[i]*u_new_SOR[i+1]-2*u_new_SOR[i])
       
        norm_SOR = abs(max(Error_SOR, key=abs))
        itererr_SOR.append((iter_sor,norm_SOR))
       
        if norm_SOR < 0.00000001:                                                        # convergence criteria
          converged = True

        for i in range(1,num_mesh-1):
            SOR1[i] = u_new_SOR[i]
        iter_sor = iter_sor+1
                # plot the converged results
plt.plot(xmesh,SOR1,'r-o')
plt.plot(xmesh,exact,'g-o')
plt.xlabel('Mesh Points')
plt.ylabel('Solution')
plt.show()
print('Number of iterations for SOR = ')
print(iter_sor)

J2=np.zeros(num_mesh)
S2=np.zeros(num_mesh)
SOR2=np.zeros(num_mesh)

J2= abs(max(J1-exact,key=abs))
S2=abs(max(S1-exact,key=abs))
SOR=abs(max(SOR1-exact,key=abs))

acc_order = open("OOA.txt", "a") #initialising the opening of a file to save error results for future use
str_num_mesh = str(num_mesh-1)
str_norm_J = str(J2)
str_norm_S = str(S2)
str_norm_SOR = str(SOR2)
acc_order.write(str_num_mesh + '\t' + str_norm_J + '\t' + str_norm_S + '\t' + str_norm_SOR + '\n') #saving the error results in a text file
acc_order.close() #closing the file

norm_J=[0.053029271854155624 , 0.012950685794714145 , 0.0032187060854194094, 0.0008025415390473523 , 0.0001966717328314438 ,3.360194397150629e-05]
norm_S= [0.05302927807494173 , 0.012950692305329747 , 0.0032187131124392643 ,0.0008025535712101739, 0.00019667882705842743 , 3.360903959426231e-05]
norm_SOR = [0.05302927293981785, 0.012950753257206182,0.0032187857481678073 , 0.0008026202504471946 , 0.00019675429276766465 ,3.3690880978709536e-05]



mesh_intervals = [4,8,16,32,64,128]
fir = np.zeros(len(mesh_intervals))
sec = np.zeros(len(mesh_intervals))
thi = np.zeros(len(mesh_intervals))
fou = np.zeros(len(mesh_intervals))
x_mesh_intervals = np.zeros(len(mesh_intervals))
for i in range(0,len(mesh_intervals)):
    x_mesh_intervals[i] = 1/mesh_intervals[i]
    fir[i] = x_mesh_intervals[i]
    sec[i] = x_mesh_intervals[i]**2
    thi[i] = x_mesh_intervals[i]**3
    fou[i] = x_mesh_intervals[i]**4
plt.plot(x_mesh_intervals,norm_J, label='Jacobi norm', marker='v', color='red')
plt.plot(x_mesh_intervals,norm_S, label='Seidel norm', marker='^', color='blue')
plt.plot(x_mesh_intervals,norm_SOR, label='SOR norm', marker='s', color='black')
plt.plot(x_mesh_intervals,fir, label='First order', linestyle='dashed', color='lightsalmon', marker='s')
plt.plot(x_mesh_intervals,sec, label='Second order', linestyle='dashed', color='darksalmon')
plt.plot(x_mesh_intervals,thi, label='Third order', linestyle='dashed', color='chocolate')
plt.plot(x_mesh_intervals,fou, label='Fourth order', linestyle='dashed', color='sienna')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\Delta x$')
plt.ylabel('L-infinity norm value')
plt.legend()
plt.show()
   
plt.plot([i[0] for i in itererr_J],[i[1] for i in itererr_J], label= 'Jacobi iteration error', color='green')
plt.plot([i[0] for i in itererr_S],[i[1] for i in itererr_S], label= 'Seidel iteration error', color='red')
plt.plot([i[0] for i in itererr_SOR],[i[1] for i in itererr_SOR], label= 'SOR iteration error', color='blue')
plt.yscale('log')
plt.xlabel('Number of iteration')
plt.legend()
plt.show()
