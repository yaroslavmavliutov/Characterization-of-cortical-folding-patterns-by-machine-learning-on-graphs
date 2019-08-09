class pitsgraph:
    def __init__(self, A, X, D, Y, S=None, T=None, other_coords=None):
        # adjacancy matrix
        self.A = A

        # coordinates of each pit
        self.X = X

        # attributes of the pits: depth
        self.D = D

        # labels of the pits: 0 or 1
        self.Y = Y

        # attributes of the basins: area (surface)
        self.S = S

        # attributes of the basins: mean thickness
        self.T = T

        # other coordinates just in case (here, the spherical coordinates, rho & theta)
        self.other_coords = other_coords
