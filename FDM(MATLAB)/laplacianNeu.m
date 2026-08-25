function A = laplacianNeu(Nx, Ny, dx, dy, dt, cx, cy)

sx=cx*dt/dx^2;
sy=cy*dt/dy^2;

ex = ones(Nx,1);
ey = ones(Ny,1);

Tx = spdiags([sx*ex -2*sx*ex sx*ex], [-1 0 1], Nx, Nx);
Ty = spdiags([sy*ey -2*sy*ey sy*ey], [-1 0 1], Ny, Ny);

% Fix boundaries (truncate stencil)
Tx(1,1)   = -sx;   Tx(Nx,Nx) = -sx;
Ty(1,1)   = -sy;   Ty(Ny,Ny) = -sy;

A = kron(speye(Ny), Tx) + kron(Ty, speye(Nx));

end