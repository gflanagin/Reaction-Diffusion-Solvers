function cost = evalcost(S,E,I,R,V,u,dM,Nt, Nx, Ny,T0,dt,dx,dy,d,mu,beta,sigma,rho,c0,c1,c2,c3,c4,epsilon,gamma1,gamma2,delta,k)


cost = .5*dt*(c0*I + c1*sigma*rho*E + c2*squeeze(u(1,:,:).*u(1,:,:)) + c3*squeeze((u(1,:,:).*u(1,:,:)+epsilon).^(1/2)));
for t=1:Nt
    ut = squeeze(u(t,:,:));
    
    time = T0+(t-1)*dt;
    b = bf(time);

    S1 = dM \ reshape(S + dt*(b*(S+E+R) - d*S - beta*S.*I - (V./(V+k)).*S), Nx*Ny,1);
    E1 = dM \ reshape(E + dt*(beta*S.*I - (sigma + d)*E), Nx*Ny, 1);
    I1 = dM \ reshape(I + dt*(sigma*rho*E - mu*I), Nx*Ny,1);
    R1 = dM \ reshape(R + dt*(sigma*(1-rho)*E - d*R + (V./(V+k)).*S), Nx*Ny,1);
    V1 = V + dt*(ut - V.*(gamma1*(S+E+R) + gamma2*I + delta));

    S = reshape(S1,Nx,Ny);
    E = reshape(E1,Nx,Ny);
    I = reshape(I1,Nx,Ny);
    R = reshape(R1,Nx,Ny);
    V = V1;
    
    if t==Nt
        cost = cost + .5*dt*(c0*I + c1*sigma*rho*E + c2*squeeze(u(t+1,:,:).*u(t+1,:,:)) + c3*squeeze((u(t+1,:,:).*u(t+1,:,:)+epsilon).^(1/2)));
    else
        cost = cost + dt*(c0*I + c1*sigma*rho*E + c2*squeeze(u(t+1,:,:).*u(t+1,:,:)) + c3*squeeze((u(t+1,:,:).*u(t+1,:,:)+epsilon).^(1/2)));
    end
end

cost = (dx/2)*(squeeze(cost(1,:)) + 2*sum(cost(2:Nx-1,:)) + squeeze(cost(Nx,:)));
cost = (dy/2)*(cost(1) + 2*sum(cost(2:Ny-1)) + cost(Ny));

terminal_int = (dx/2)*(squeeze(I(1,:)) + 2*sum(I(2:Nx-1,:)) + I(Nx,:));
terminal_int = (dy/2)*(terminal_int(1) + 2*sum(terminal_int(2:Ny-1)) + terminal_int(Ny));

cost = cost + c4*terminal_int;

end