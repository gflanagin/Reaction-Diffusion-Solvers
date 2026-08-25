% intrinsic model parameters
d=.026; mu=.490; beta=.04; sigma=2; rho=1;
defaultdiffusion=.5; forestdiffusion=.2; riverdiffusion=0;
defaultdensity=30; forestdensity=10; riverdensity=0;
defaultcapacity=30; forestcapacity=10;
outbreakdensity=10;

%vaccination and cost parameters
k=1; gamma1=.01; gamma2=.01; delta=.01;
c0=0; c1=1; c2=0; c3=30; c4=10; epsilon=.00000001;


outbreak = {[3 7; 2 4]};
forest = {[0 4; 16 20],[7 20; 0 10],[15 20; 10 15]};
river = {[7 7.3; 10 20],[11 11.3; 0 10],[0 8; 8 8.3]};


%mesh parameters
T0 = 10;
Tf = 52;
dt=.25;
Nt = (Tf-T0)/dt;
Nx=151;
dx=20/150;
Ny=151;
dy=20/150;
Nr=70;

%control and line search parameters
u = zeros(Nt+1,Nx,Ny);
tolerance = .01;
alpha = 1;

%initialize I and L to incorporate continuum coeeficients a_1,a_2.
S0 = defaultdensity*ones(Nx,Ny);
E0 = zeros(Nx,Ny);
I0 = zeros(Nx,Ny);
R0 = zeros(Nx,Ny);
V0 = zeros(Nx,Ny);
L=laplacianNeu(Nx,Ny,dx,dy,dt, defaultdiffusion, defaultdiffusion);
for i = 1:Nx
    for j = 1:Ny
        zone = false;
        for k = 1:length(outbreak)
            if (outbreak{k}(1,1) <= i*dx) && (i*dx <= outbreak{k}(1,2)) && (outbreak{k}(2,1) <= j*dy) && (j*dy <= outbreak{k}(2,2))
                I0(i,j) = outbreakdensity;
                zone =  true;
            end
        end
        
        for k = 1:length(forest)
            if (forest{k}(1,1) <= i*dx) && (i*dx <= forest{k}(1,2)) && (forest{k}(2,1) <= j*dy) && (j*dy <= forest{k}(2,2))
                S0(i,j) = forestdensity;
                if zone == true
                    I0(i,j) = outbreakdensity;
                end
                
                %Changes diffusion coefficients
                sigmax = forestdiffusion*dt/dx^2;
                sigmay = forestdiffusion*dt/dy^2;
                L = changediff(L,i,j,Nx,Ny,sigmax,sigmay);
            end
        end
        
        for k=1:length(river)
            if (river{k}(1,1) <= i*dx) && (i*dx <= river{k}(1,2)) && (river{k}(2,1) <= j*dy) && (j*dy <= river{k}(2,2))
                S0(i,j) = riverdensity;
                if zone == true
                    I0(i,j) = riverdensity;
                end
    
                %Change diffusion coefficients
                sigmax = riverdiffusion*dt/dx^2;
                sigmay = riverdiffusion*dt/dy^2;
                L = changediff(L,i,j,Nx,Ny,sigmax,sigmay);
            end
        end
    end
end

%initialize the stiffness matrix
M = (speye(Nx*Ny)-L);
dM = decomposition(M);

% Create single figure with 3 panels
fig = figure;
set(fig,'Name','SIR Simulation')
set(fig,'Position',[0 0 1800 1800]) 

[X,Y] = meshgrid((1:Nx)*dx,(1:Ny)*dy);

insteps = T0/dt;
[S0,E0,I0,R0] = initialize(S0,E0,I0,R0,dM,insteps, Nx, Ny,dt,d,mu,beta,sigma,rho);

disp('initialized')

%% --- Susceptible surface ---
subplot(2,3,1)
Ssurf = surf(X,Y,S0');
shading interp
colorbar
clim([0 30])
zlim([0 60])
view(45,30)
title('Susceptible: time 0')

%% --- Infected(not infectious) surface ---
subplot(2,3,2)
Esurf = surf(X,Y,E0');
shading interp
colorbar
clim([0 10])
zlim([0 20])
view(45,30)
title('Infected(not infectious): time 0')

%% --- Infectious surface ---
subplot(2,3,3)
Isurf = surf(X,Y,I0');
shading interp
colorbar
clim([0 15])
zlim([0 30])
view(45,30)
title('Infectious: time 0')

%% --- Immune surface ---
subplot(2,3,4)
Rsurf = surf(X,Y,R0');
shading interp
colorbar
clim([0 15])
zlim([0 30])
view(45,30)
title('Infectious: time 0')

%% --- Vaccine Bait surface --
subplot(2,3,5)
Vsurf = surf(X,Y,V0');
shading interp
colorbar
clim([0 .1])
zlim([0 .3])
view(45,30)
title('Vaccine baits: time 0')


%% --- Control surface ---
subplot(2,3,6)
Csurf = surf(X,Y,squeeze(u(1,:,:))');
shading interp
colorbar
clim([0 .05])
zlim([0 .15])
view(45,30)
title('Control: time 0')


%initialize video
v = VideoWriter('raccoon_control.avi');
v.FrameRate = 30;   % adjust playback speed if desired
open(v);
%% --- Main loop ---

S=zeros(Nt+1,Nx,Ny);
E=zeros(Nt+1,Nx,Ny);
I=zeros(Nt+1,Nx,Ny);
R=zeros(Nt+1,Nx,Ny);
V=zeros(Nt+1,Nx,Ny);

S(1,:,:) = S0;
E(1,:,:) = E0;
I(1,:,:) = I0;
R(1,:,:) = R0;
V(1,:,:) = V0;


for j=1:Nr

    cost=zeros(Nx,Ny);

    for t=1:Nt
   
        ut = squeeze(u(t,:,:));
        St = squeeze(S(t,:,:));
        Et = squeeze(E(t,:,:));
        It = squeeze(I(t,:,:));
        Rt = squeeze(R(t,:,:));  
        Vt = squeeze(V(t,:,:));

        time = T0+(t-1)*dt;
        b = bf(time);

        S1 = dM \ reshape(St + dt*(b*(St+Et+Rt) - d*St - beta*St.*It - (Vt./(Vt+k)).*St), Nx*Ny,1);
        E1 = dM \ reshape(Et + dt*(beta*St.*It - (sigma + d)*Et), Nx*Ny, 1);
        I1 = dM \ reshape(It + dt*(sigma*rho*Et - mu*It), Nx*Ny,1);
        R1 = dM \ reshape(Rt + dt*(sigma*(1-rho)*Et - d*Rt + (Vt./(Vt+k)).*St), Nx*Ny,1);
        V1 = Vt + dt*(ut - Vt.*(gamma1*(St+Et+Rt) + gamma2*It) - Vt*delta);
    
        S(t+1,:,:) = reshape(S1,Nx,Ny);
        E(t+1,:,:) = reshape(E1,Nx,Ny);
        I(t+1,:,:) = reshape(I1,Nx,Ny);
        R(t+1,:,:) = reshape(R1,Nx,Ny);
        V(t+1,:,:) = V1;
    
        if t==1
            cost = cost + (dt/2)*(c0*It + c1*sigma*rho*Et + c2*(ut.*ut) + c3*(ut.*ut+epsilon).^(1/2));
        else
            cost = cost + dt*(c0*It + c1*sigma*rho*Et + c2*(ut.*ut) + c3*(ut.*ut+epsilon).^(1/2));
        end      
       
        
        if j==Nr
            if mod(t,4)==0
        
                % --- update surfaces ---
                set(Ssurf,'ZData',squeeze(S(t+1,:,:))','CData',squeeze(S(t+1,:,:))');
                set(Esurf,'ZData',squeeze(E(t+1,:,:))','CData',squeeze(E(t+1,:,:))');
                set(Isurf,'ZData',squeeze(I(t+1,:,:))','CData',squeeze(I(t+1,:,:))');
                set(Rsurf,'ZData',squeeze(R(t+1,:,:))','CData',squeeze(R(t+1,:,:))');
                set(Vsurf,'ZData',squeeze(V(t+1,:,:))','CData',squeeze(V(t+1,:,:))');
                set(Csurf,'Zdata', squeeze(u(t+1,:,:))', 'CData', squeeze(u(t+1,:,:))');
            
                %update titles
                subplot(2,3,1)
                title(sprintf('Susceptible (Week %.2f)',T0+t*dt))
            
                subplot(2,3,2)
                title(sprintf('Infected(not infectious) (Week %.2f)',T0+t*dt))
    
                subplot(2,3,3)
                title(sprintf('Infectious (Week %.2f)',T0+t*dt))
    
                subplot(2,3,4)
                title(sprintf('Immune (Week %.2f)',T0+t*dt))
    
                subplot(2,3,5)
                title(sprintf('Vaccine baits (Week %.2f)',T0+t*dt))
        
                subplot(2,3,6)
                title(sprintf('Control (Week %.2f)',T0+t*dt))
                
                drawnow
                frame = getframe(fig);
                writeVideo(v,frame);
            end
        end
    end
    
    It = squeeze(I(Nt+1,:,:));
    Et = squeeze(E(Nt+1,:,:));
    ut = squeeze(u(Nt+1,:,:));
    
    cost = cost + (dt/2)*(c0*It + c1*sigma*rho*Et + c2*(ut.*ut) + c3*(ut.*ut+epsilon).^(1/2));
    cost = (dx/2)*(squeeze(cost(1,:)) + 2*sum(cost(2:Nx-1,:)) + squeeze(cost(Nx,:)));
    cost = (dy/2)*(cost(1) + 2*sum(cost(2:Ny-1)) + cost(Ny));

    terminal_int = (dx/2)*(squeeze(It(1,:)) + 2*sum(It(2:Nx-1,:)) + It(Nx,:));
    terminal_int = (dy/2)*(terminal_int(1) + 2*sum(terminal_int(2:Ny-1)) + terminal_int(Ny));
    
    cost = cost + c4*terminal_int;

    disp('cost:')
    disp(cost)

    if j ~= Nr
        gradH = backsolve(S,E,I,R,V,u,dM,dt,dx,dy,Nt,Nx,Ny,Tf,d,mu,beta,sigma,rho,c0, c1,c2,c3,c4,epsilon,gamma1,gamma2,delta,k);
        normg = norm(gradH, 'fro');
        disp('grad norm:');
        disp(normg);
        u = armijo(S0,E0,I0,R0,V0,tolerance,alpha,u,cost,gradH,dM,Nt,Nx,Ny,T0,dt,dx,dy,d,mu,beta,sigma,rho,c0,c1,c2,c3,c4,epsilon,gamma1,gamma2,delta,k);
        %u = u - .05*gradH;
        %u = min(1,max(0,u));
        disp(j)
        %tolerance = .05;
    end
end
close(v);


