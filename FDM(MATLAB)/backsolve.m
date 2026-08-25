function gradH = backsolve(S,E,I,R,V,u,dM,dt,dx,dy,Nt, Nx, Ny,Ft,d,mu,beta,sigma,rho,c0,c1,c2,c3,c4,epsilon,gamma1,gamma2, delta,k)

lam1 = zeros(Nx,Ny);
lam2 = zeros(Nx,Ny);
lam3 = c4*ones(Nx,Ny);
lam4 = zeros(Nx,Ny);
lam5 = zeros(Nx,Ny);
gradH = zeros(Nt+1,Nx,Ny);

for t=1:Nt
    ut = squeeze(u(Nt+2-t,:,:));
    St = squeeze(S(Nt+2-t,:,:));
    Et = squeeze(E(Nt+2-t,:,:));
    It = squeeze(I(Nt+2-t,:,:));
    Rt = squeeze(R(Nt+2-t,:,:));
    Vt = squeeze(V(Nt+2-t,:,:));
    
    gradH(Nt+2-t,:,:) = dx*dy*dt*(2*c2*ut + c3*ut.*(ut.*ut+epsilon).^(-1/2) + lam5);
    b = bf(Ft-dt*(t-1));

    lam11 = dM \ reshape(lam1 + dt*(lam1*(b-d) + beta*It.*(lam2-lam1) + (Vt./(Vt+k)).*(lam4-lam1) - gamma1*Vt.*lam5),Nx*Ny,1);
    lam21 = dM \ reshape(lam2 + dt*(c1*sigma*rho + b*lam1 - d*lam2 + sigma*rho*(lam3-lam2) + sigma*(1-rho)*lam4 - gamma1*Vt.*lam5), Nx*Ny, 1);
    lam31 = dM \ reshape(lam3 + dt*(c0 + beta*St.*(lam2-lam1) - mu*lam3 - gamma2*Vt.*lam5), Nx*Ny, 1);
    lam41 = dM \ reshape(lam4 + dt*(b*lam1 - d*lam4 - gamma1*Vt.*lam5), Nx*Ny, 1);
    lam51 = lam5 + dt*(k*St.*(1./(Vt+k)).*(1./(Vt+k)).*(lam4-lam1) - lam5.*(gamma1*(St+Et+Rt) + gamma2*It + delta));

    lam1 = reshape(lam11, Nx, Ny);
    lam2 = reshape(lam21, Nx, Ny);
    lam3 = reshape(lam31, Nx, Ny);
    lam4 = reshape(lam41, Nx, Ny);
    lam5 = lam51;

end

ut = u(Nt+1,:,:);
gradH(1,:,:) = dx*dy*dt*(squeeze(2*c2*ut + c3*ut.*(ut.*ut+epsilon).^(-1/2)) + lam5);

end