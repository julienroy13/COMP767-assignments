import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
import pdb

class SimpleOptim(Optimizer):
    """
    """

    def __init__(self, params, lr, decay, discount):
        defaults = dict(lr=lr, decay=decay, discount=discount)
        if lr <= 0:
            raise ValueError("learning rate (step size) must be strickly positive.")
        if not (0 <= decay <= 1):
            raise ValueError("trace-decay rate must be between 0 and 1.")
        super(SimpleOptim, self).__init__(params, defaults)
        
        #
        self.eligibilities  = dict()
        self.I = 1
        
        # initialise eligibilities ([re]set to zero(0))
        self._reset_eligibilities()

    def __setstate__(self, state):
        super(SimpleOptim, self).__setstate__(state)

    def step(self, error, closure=None):
        """
        """
        loss = None
        if closure is not None:
            loss = closure()
                
        # update parameters  
        for i, group in enumerate(self.param_groups):
            
            #
            _delta = error
            
            # parameters: trace-decay rates for policy and state-value,
            #             step sizes for policy and state-value,
            _alpha = group["lr"]
            _lambda = group["decay"]
            _gamma = group["discount"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                # retrieve current eligibility
                z = self.eligibilities[i][p]
                # retrieve current gradient
                grad = p.grad.data
                # update eligibility
                z.mul_(_gamma * _lambda).add_(self.I, grad)
                # update parameters
                p.data.add_(_alpha * _delta * z)
                
        #
        self.I = _gamma * self.I
    
    def _reset_eligibilities(self):
        for i, group in enumerate(self.param_groups):
            zs = dict()
            for p in group["params"]:
                zs[p] = torch.zeros_like(p.data)
            self.eligibilities[i] = zs
            
            
class ActorCritic(object):
    def __init__(self, actor, critic, discount, lr_critic, lr_actor, decay_critic, decay_actor, use_cuda, gpu_id):
        
        self.use_cuda = use_cuda
        self.gpu_id = gpu_id

        # models: a differentiable policy parameterization (actor),
        #         a differentiable state-value parameterization (critic).
        self.actor  = actor
        self.critic = critic
        
        # push to GPU device if available
        if torch.cuda.is_available() and self.use_cuda:
            self.critic = self.critic.cuda(gpu_id)
            self.actor  = self.actor.cuda(gpu_id)
        
        # parameters: trace-decay rates for policy and state-value,
        #             step sizes for policy and state-value,
        #             discount factor.
        
        if not (0 <= discount <= 1):
            raise ValueError("discount factor must be between 0 and 1.")
        self._gamma         = discount
        
        #
        self.critic_optimizer = SimpleOptim(critic.parameters(), lr_critic, decay_critic, discount)
        self.actor_optimizer  = SimpleOptim(actor.parameters(), lr_actor, decay_actor, discount)
        
        #
        self.state = None
        
        
    def select_action(self, observation):
        
        # to torch.autograd.Variable if necessary
        state = self._convert(observation)
            
        # sample action from policy
        probas = self.actor(state)
        action = int(probas.multinomial().data)
        
        return action
    
    def compute_gradients(self, action):
        # brackward propagation for state-value function
        v = self.critic(self.state)
        v.backward()
        # backward propagation for log-probability of the selected action
        probas = self.actor(self.state)
        log_proba = torch.log(probas)[:, action]
        log_proba.backward()
    
    def update_parameters(self, observation, reward):
        
        # to torch.autograd.Variable if necessary
        obs = self._convert(observation)
            
        # compute TD error
        error = reward + (self._gamma * self.critic(obs)) - self.critic(self.state)
        _delta = error.cpu().data
        # update critic's parameters
        self.critic_optimizer.step(error=_delta)
        self.critic_optimizer.zero_grad()
                
        # update actor's parameters     
        self.actor_optimizer.step(error=_delta)
        self.actor_optimizer.zero_grad()
    
    def set_state(self, observation):
        # to torch.autograd.Variable if necessary
        state = self._convert(observation)
        # save state
        self.state = state
    
    def _convert(self, x):
        # to torch.autograd.Variable if necessary
        if not isinstance(x, Variable):
            s = Variable(torch.Tensor([x]))  
            if torch.cuda.is_available() and self.use_cuda:
                s = s.cuda(gpu_id)
        else:
            s = x
        return s