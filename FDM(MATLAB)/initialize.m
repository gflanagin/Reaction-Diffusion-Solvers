function [S,E,I,R] = initialize(S,E,I,R,dM,Nt, Nx, Ny,dt,d,mu,beta,sigma,rho)

for t=1:Nt
    b = bf((t-1)*dt);

    S1 = dM \ reshape(S + dt*(b*(S+E+R) - d*S - beta*S.*I), Nx*Ny,1);
    E1 = dM \ reshape(E + dt*(beta*S.*I - (sigma + d)*E), Nx*Ny, 1);
    I1 = dM \ reshape(I + dt*(sigma*rho*E - mu*I), Nx*Ny,1);
    R1 = dM \ reshape(R + dt*(sigma*(1-rho)*E - d*R), Nx*Ny,1);

    S = reshape(S1,Nx,Ny);
    E = reshape(E1,Nx,Ny);
    I = reshape(I1,Nx,Ny);
    R = reshape(R1,Nx,Ny);
end
end