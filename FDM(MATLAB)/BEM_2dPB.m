N = 10; k=1;

M=zeros(N,N);
for j=1:N
    for i=1:N
      if i==j
          M(i,j)=-1/2;
      else
          xi = [cos(2*pi*i/N) sin(2*pi*i/N)];
          xj = [cos(2*pi*j/N) sin(2*pi*j/N)];
          r = norm(xi-xj,2);
          M(i,j)=(-k/N)*besselk(1,k*r)*dot(xj-xi,xj)/r;
      end
    end
end

disp(det(M))
Bvalues = M \ zeros(N,1);