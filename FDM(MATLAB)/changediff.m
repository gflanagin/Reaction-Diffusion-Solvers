function L = changediff(L,i,j,Nx,Ny,sigmax,sigmay)

index = i+Nx*(j-1);
L(index,index) = 0;
if i ~= 1
    L(index,index) = L(index,index) - sigmax;
    L(index,index-1) = sigmax;
    
    L(index-1,index-1) = L(index-1,index-1) + L(index-1,index) - sigmax;
    L(index-1,index) = sigmax;
end
if i ~= Nx
    L(index,index) = L(index,index) - sigmax;
    L(index,index+1) = sigmax;

    L(index+1,index+1) = L(index+1,index+1) + L(index+1,index) - sigmax;
    L(index+1,index) = sigmax; 
end
if j ~= 1
    L(index,index) = L(index,index) - sigmay;
    L(index,index-Nx) = sigmay;

    L(index-Nx,index-Nx) = L(index-Nx,index-Nx) + L(index-Nx,index) - sigmay;
    L(index-Nx,index) = sigmay;
end
if j ~= Ny
    L(index,index) = L(index,index) - sigmay;
    L(index,index+Nx) = sigmay;
    
    L(index+Nx,index+Nx) = L(index+Nx,index+Nx) + L(index+Nx,index) - sigmay;
    L(index+Nx,index) = sigmay;
end