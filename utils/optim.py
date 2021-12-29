'''A wrapper class for optimizer '''


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.warmup = n_warmup_steps ** -1.5
        self.n_current_steps = 0
        self.init_lr = d_model ** -0.5
        # self.init_lr = 0.02
        print('warm up steps: ', n_warmup_steps)
        print('d_model: ', d_model)
        print('optimizer: ', optimizer)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return min(self.n_current_steps ** -0.5, self.warmup * self.n_current_steps)

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
