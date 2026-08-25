import meshio

msh = meshio.read("terrain.msh")

# extract surface triangles
triangles = msh.get_cells_type("triangle")

meshio.write(
	"surface.xdmf",
	meshio.Mesh(
		points=msh.points,
		cells={"triangle": triangles},
	)
)