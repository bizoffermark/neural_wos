import torch


class BasePDEEquation:
    """
    A class for PDE equations
    """

    def __init__(
        self,
        n_dim: int,
        stop_tol: float,
    ) -> None:
        """
        We assume that the PDE is of the form:
            PDE[u(x)] = 0 if domain_test_fn(x) = True => PDE condition
            u(x) = bound_fn(x) if bound_test_fn(x) = True => boundary condition

        """
        self.stop_tol = stop_tol
        self.n_dim = n_dim  # dimension of the problem
        # if domain_test_fn is not None:
        #     self.domain_test_fn = domain_test_fn
        # if domain_sample_fn is not None:
        #     self.domain_sample_fn = domain_sample_fn
        # if bound_test_fn is not None:
        #     self.bound_test_fn = bound_test_fn
        # if get_closest_bound_fn is not None:
        #     self.get_closest_bound_fn = get_closest_bound_fn
        # if bound_fn is not None:
        #     self.bound_fn = bound_fn

    def domain_test_fn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Test if the points are in the domain
        """
        raise NotImplementedError

    def domain_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample points from the domain
        """
        raise NotImplementedError

    def domain_fn(self, x: torch.Tensor) -> torch.Tensor:
        """
        The function at the domain points
        """
        raise NotImplementedError

    def bound_test_fn(
        self, x: torch.Tensor | None = None, d: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Test if the points are on the boundary
        """
        if d is None:
            d = self.get_closest_bound_fn(x)[1]
        cond = d < self.stop_tol
        return cond

    def get_closest_bound_fn(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Get the closest boundary point
        """
        raise NotImplementedError

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        """
        The boundary function at the boundary points
        """
        raise NotImplementedError

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample points from the boundary
        """
        raise NotImplementedError

    @staticmethod
    def get_true_solution_fn(x: torch.Tensor) -> torch.Tensor:
        """
        True analytical function for the PDE (if available)
        """
        pass
