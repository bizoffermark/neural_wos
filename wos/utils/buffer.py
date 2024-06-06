from typing import List
import torch


class TrainBuffer:
    '''
        Basically the same as DynamicBuffer except this one generalizes it and hopefully a better version of it !!
    '''
    def __init__(
        self,
        n_sample: int,
        max_step: int,
        n_traj: int,
        n_traj_init: int, # initial trajectory to fill the buffer
        n_dim: int,
        device: torch.device,
        bound_init: bool,
        priority_evict: bool = False,
        priority_walk: bool = False,
    ):
        # size of the buffer
        self.n_sample = n_sample
        # maximum number of steps
        self.max_step = max_step
        # number of trajectories
        self.n_traj = n_traj
        # number of initial trajectories to fill the buffer
        self.n_traj_init = n_traj_init
        # whether to use the boundary points to initialize the buffer
        self.bound_init = bound_init

        # dimension of the space
        self.n_dim = n_dim
        # device
        self.device = device

        # priority eviction
        self.priority_evict = priority_evict
        self.priority_walk = priority_walk

        # mask to indicate if each point lies in the boundary or no
        self.mask = torch.zeros(n_sample,
                                device=device,
                                dtype=torch.bool)

        # walk counter
        self.counter = torch.zeros(
            self.n_sample, device=self.device, dtype=torch.int64
        )

        # grid points
        self.x = torch.empty(
            [self.n_sample, self.n_dim], device=self.device
        )

        # final value f(x)
        self.y = torch.empty(
            [self.n_sample, 1], device=self.device
        )

        # walked frequency
        self.freq = torch.zeros(
            [self.n_sample, 1], device=self.device, dtype=torch.int64
        )

        self.full = False
        self.n_domain_pts = 0 # number of points in the domain

    def domain_percentage(self):
        '''
            Return the percentage of points in the buffer that are in the domain
        '''
        return self.mask.sum()/self.n_sample
    
    def average_walked(self):
        '''
            Return the average number of times each point is walked
        '''
        # get 25%, 50%, 75%, 90% percentiles, mean and max
        
        float_freq = self.freq.float()
        report = {
            'mean': float_freq.mean().item(),
            'min': float_freq.min().item(),
            '50%': float_freq.median().item(),
            '25%': float_freq.quantile(0.25).item(),
            '75%': float_freq.quantile(0.75).item(),
            '90%': float_freq.quantile(0.90).item(),
            'max': float_freq.max().item()
        }
        return report

    def fill(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        buffer_index: torch.Tensor = None,
        domain_indicator: torch.Tensor = None,  # indicator => True if in the domain
    ) -> None:
        """
        Fill the tensor list 

        """
        assert x.shape[-1] == self.n_dim
        assert y.shape[-1] == 1 
        assert x.device.type == self.device
        assert y.device.type == self.device
        if self.full:
            # Buffer is already full
            assert domain_indicator is not None
            if buffer_index is None:
                in_buffer_indicator = torch.zeros_like(domain_indicator, dtype=torch.bool, device=self.device)
                out_buffer_indicator = torch.ones_like(domain_indicator, dtype=torch.bool, device=self.device)
            else:
                in_buffer_indicator = buffer_index < self.n_sample
                out_buffer_indicator = buffer_index >= self.n_sample
            in_update_indicator = domain_indicator & in_buffer_indicator
            if in_update_indicator.sum() > 0:
                freq = self.freq[buffer_index[in_update_indicator]]
                self.y[buffer_index[in_update_indicator]] = freq/(freq+self.n_traj) * self.y[buffer_index[in_update_indicator]] + self.n_traj/(freq+self.n_traj) * y[in_update_indicator]
                self.freq[buffer_index[in_update_indicator]] += self.n_traj
            del in_update_indicator

            out_update_indicator = out_buffer_indicator & domain_indicator
            if out_update_indicator.sum() > 0:
                # points not on the buffer are NOT always in the domain!
                random_index = self.reset_partial(out_update_indicator.sum())
                self.x[random_index] = x[out_update_indicator]
                self.y[random_index] = y[out_update_indicator]
                self.freq[random_index] = self.n_traj
                self.mask[random_index] = torch.ones_like(self.mask[random_index], dtype=torch.bool, device=self.device)
                del out_update_indicator
        else:
            # buffer is empty
            assert x.shape[0] == self.n_sample
            self.x = x
            self.y = y
            self.freq[domain_indicator] += self.n_traj_init # initial fill
            self.mask = domain_indicator
            self.full = True
            
        self.n_domain_pts = (self.freq > 0).sum().item()

    def reset(self):
        """
        Clean the tensor list
        """

        # grid points
        self.x = torch.empty(
            [self.n_sample, self.n_dim], device=self.device
        )

        # final value f(x)
        self.y = torch.empty(
            [self.n_sample, 1], device=self.device
        )

        # walked frequency
        self.freq = torch.zeros(
            [self.n_sample, 1], device=self.device, dtype=torch.int64
        )

        self.full = False

    def sample(self,
               n_batch: int,
               y_true: torch.Tensor=None,
               train_phase: bool = True,
               ) -> List[torch.Tensor]:
        """
        Sample points from the buffer
        """
        assert self.full # only sample AFTER the buffer is full
        if y_true is None:
            if train_phase:
                # when training, we sample uniformly from the buffer
                buffer_index = torch.multinomial(torch.ones(self.n_sample, device=self.device), n_batch, replacement=False)
            else:
                # when walking, we have 2 possible ways to smaple
                if self.n_domain_pts < n_batch:
                    # we have less points in the buffer than n_batch => return all of them
                    buffer_index = self.freq[self.freq > 0].squeeze(-1)
                else:
                    if self.priority_walk:
                        buffer_index = torch.multinomial(1/self.freq.squeeze(-1).float(), n_batch, replacement=False)
                    else:
                        buffer_index = torch.multinomial(torch.ones(self.n_sample, device=self.device), n_batch, replacement=False)
            return self.x[buffer_index], self.y[buffer_index], buffer_index, self.mask[buffer_index]
        else:
            # since true solution is known, we can sample based on the error
            # ONLY DONE FOR DEBUGGING
            assert y_true.shape == self.y.shape
            buffer_index = torch.topk(((y_true-self.y)**2).squeeze(-1),
                                      n_batch, largest=True).indices
            return self.x[buffer_index], self.y[buffer_index], buffer_index, self.mask[buffer_index]


    def reset_partial(self,
                      n_batch: int
                      ) -> torch.Tensor:
        '''
            Reset some of the points in the buffer
            We do priority sampling here where the ones that are walked multiple times 
        '''

        assert self.full
        # use priority sampling by choosing ones with small frequency more often
        if self.priority_evict:
            # reset ones with small frequency more often
            idx = torch.multinomial(1/self.freq.squeeze(-1).float(), n_batch, replacement=False) 
        else:
            idx = torch.multinomial(torch.ones(self.n_sample, device=self.device), n_batch, replacement=False)
        self.freq[idx] = 0
        return idx
