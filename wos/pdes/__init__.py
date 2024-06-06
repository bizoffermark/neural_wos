# define a dict to store all PDEs

from wos.pdes.base_pde import BasePDEEquation
from wos.pdes.committor_nd import CommittorNd
from wos.pdes.laplace_2d_annulus import LaplaceAnnulus
from wos.pdes.laplace_nd import LaplaceNd
from wos.pdes.poisson_2d import Poisson2d
from wos.pdes.poisson_2d_mc import Poisson2d_MC
# from wos.pdes.poisson_2d_test import Poisson2d_test
from wos.pdes.poisson_nd import PoissonNd

pdes_dict = {
    "base_pde": BasePDEEquation,
    "laplace_annulus": LaplaceAnnulus,
    "poisson_2d": Poisson2d,
    "laplace_nd": LaplaceNd,
    # "poisson_2d_test": Poisson2d_test,
    "poisson_nd": PoissonNd,
    "committor_nd": CommittorNd,
    "poisson_2d_mc": Poisson2d_MC,
}


# def get_pde(cfg: PDEParams):
#     """
#     Get the PDE
#     """
#     pde_name = cfg.name
#     if pde_name not in pdes_dict:
#         raise ValueError(f"Unknown PDE: {pde_name}")
#     if pde_name == "committor_nd":
#         return pdes_dict[pde_name](cfg.n_dim, cfg.a, cfg.b)
#     return pdes_dict[pde_name](cfg.n_dim)
