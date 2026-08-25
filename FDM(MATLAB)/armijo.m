function unew = armijo(S0,E0,I0,R0,V0,tolerance,alpha,u,costu,grad,dM,Nt,Nx,Ny,T0,dt,dx,dy,d,mu,beta,sigma,rho,c0,c1,c2,c3,c4,epsilon,gamma1,gamma2,delta,k)

normg = norm(grad, "fro");
m = tolerance*normg;
%disp(max(abs(grad), [], 'all'));
unew = u - alpha*grad;
unew = min(unew,1);
unew = max(unew,0);
costnew = evalcost(S0,E0,I0,R0,V0,unew,dM,Nt, Nx, Ny,T0,dt,dx,dy,d,mu,beta,sigma,rho,c0,c1,c2,c3,c4,epsilon,gamma1,gamma2,delta,k);
while costnew > costu - abs(alpha*m)
    alpha = alpha / 4;
    if abs(alpha) < .0000001
        alpha = .005;
        m = m/10;
        disp('lower tol');
    end
    unew = u - alpha*grad;
    unew = min(unew,1);
    unew = max(unew,0);
    costnew = evalcost(S0,E0,I0,R0,V0,unew,dM,Nt, Nx, Ny,T0,dt,dx,dy,d,mu,beta,sigma,rho,c0,c1,c2,c3,c4,epsilon,gamma1,gamma2,delta,k);
    disp("line search iteration");
    %disp(costnew);
    if m < .001
        disp('armijo failed');
        disp(normg);
        unew = u;
        break
    end
end

end