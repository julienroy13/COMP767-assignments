class ActorCritic(object):
    def __init__(self, actor, critic, discount, lr_critic, lr_actor, decay_critic, decay_actor):
        
        # models: a differentiable policy parameterization (actor),
        #         a differentiable state-value parameterization (critic).
        self.actor  = actor
        self.critic = critic
        
        # push to GPU device if available
        if torch.cuda.is_available():
            critic = critic.cuda()
            actor  = actor.cuda()
        
        # parameters: trace-decay rates for policy and state-value,
        #             step sizes for policy and state-value,
        #             discount factor.
        self._gamma         = discount
        self._lambda_critic = decay_critic
        self._lambda_actor  = decay_actor
        self._alpha_critic  = lr_critic
        self._alpha_actor   = lr_actor
        
        #
        critic_eligibilities = dict()
        actor_eligibilities  = dict()
        I = 1
        

        # initialise critic eligibilities ([re]set to zero(0))
        for i, group in enumerate(critic_optimizer.param_groups):
            zs = dict()
            for p in group["params"]:
                zs[p] = torch.zeros_like(p.data)
            critic_eligibilities[i] = zs

        # initialise actor eligibilities ([re]set to zero(0))
        for i, group in enumerate(actor_optimizer.param_groups):
            zs = dict()
            for p in group["params"]:
                zs[p] = torch.zeros_like(p.data)
            actor_eligibilities[i] = zs
        
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
        v = critic(state)
        v.backward()
        # backward propagation for log-probability of the selected action
        probas = actor(state)
        log_proba = torch.log(probas)[:, action]
        log_proba.backward()
        
        return None
    
    def update_parameters(self, observation):
        
        # to torch.autograd.Variable if necessary
        obs = self._convert(observation)
            
        # compute TD error
        error = reward + (_gamma * self.critic(obs)) - self.critic(self.state)
        _delta = error.cpu().data
        # update critic's parameters  
        for i, group in enumerate(critic_optimizer.param_groups):

            for p in group["params"]:
                if p.grad is None:
                    continue
                # retrieve current eligibility
                z = critic_eligibilities[i][p]
                # retrieve current gradient
                grad = p.grad.data
                # update eligibility
                z.mul_(_gamma * _lambda_critic).add_(I, grad)
                # update parameters
                p.data.add_(_alpha_critic * _delta * z)
                # reset gradients
                p.grad.detach_()
                p.grad.zero_()
                
        # update actor's parameters     
        for i, group in enumerate(actor_optimizer.param_groups):

            for p in group["params"]:
                if p.grad is None:
                    continue
                # retrieve current eligibility
                z = actor_eligibilities[i][p]
                # retrieve current gradient
                grad = p.grad.data
                # update eligibility
                z.mul_(_gamma * _lambda_actor).add_(I, grad)
                # update parameters
                p.data.add_(_alpha_actor * _delta * z)
                # reset gradients
                p.grad.detach_()
                p.grad.zero_()
        #
            self.I = _gamma * self.I
    
    def reset_counters(self, observation):
        # to torch.autograd.Variable if necessary
        state = self._convert(observation)
        # save state
        self.state = state
    
    def _convert(self, x):
        # to torch.autograd.Variable if necessary
        if not isinstance(x, Variable):
            s = Variable(torch.Tensor([x]))  
            if torch.cuda.is_available():
                s = s.cuda()
        else:
            s = x
        return s