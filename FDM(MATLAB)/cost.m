function [cost] = evalcost(S,I,R,u)

ht = dt/3;
cost = ht*(I + squeeze(u(1,:,:)));
for t=1:Nt
    ut = squeeze(u(t,:,:));

    S1 = dM \ reshape(S + dt*((b-d)*S + b*R - beta*S.*I - ut.*S), Nx*Ny,1);
    I1 = dM \ reshape(I + dt*(beta*S.*I - mu2*I), Nx*Ny,1);
    R1 = dM \ reshape(R + dt*(-d*R + ut.*S), Nx*Ny,1);

    S = reshape(S1,Nx,Ny);
    I = reshape(I1,Nx,Ny);
    R = reshape(R1,Nx,Ny);
    
    if t==Nt
        cost = cost + ht*(I + squeeze(u(t+1,:,:)));
    elseif mod(t,2)==1
        cost = cost + 4*ht*(I + squeeze(u(t+1,:,:)));
    else
        cost = cost + 2*ht*(I + squeeze(u(t,:,:)));
    end
end

cost = squeeze((dx/3)*(cost(1,:) + 4*cost(2:2:Nx-1,:) + 2*cost(3:2:Nx-2,:) + cost(Nx,:)));
cost = squeeze((dy/3)*(cost(1) + 4*cost(2:2:Ny-1) + 2*cost(3:2:Ny-2) + cost(Ny)));

end