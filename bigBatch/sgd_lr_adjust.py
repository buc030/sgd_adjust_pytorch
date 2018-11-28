import torch.optim as optim
import torch

class SGDAdjustOptimizer:

    def get_arg(self, key):
        if key not in self.optimizer_kwargs:
            return None
        return self.optimizer_kwargs[key]

    def __init__(self, base_optimizer, iters_per_adjust, **optimizer_kwargs):
        self.base_optimizer = base_optimizer
        self.iter = 0
        self.iters_per_adjust = iters_per_adjust
        self.optimizer_kwargs = optimizer_kwargs

        self.writer = self.get_arg('writer')
        self.disable_lr_change = self.get_arg('disable_lr_change')


        self.inited_buffers = False

        self.update_lr_i = self.iters_per_adjust

    def init_buffers(self):
        if self.inited_buffers == True:
            return

        self.inited_buffers = True
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue

                grad = p.data
                # grad = p.grad.data
                state = self.base_optimizer.state[p]
                # Exponential moving average of gradient values
                state['curr_snapshot'] = grad.new().resize_as_(grad).zero_()
                # Exponential moving average of squared gradient values
                state['prev_snapshot'] = grad.new().resize_as_(grad).zero_()


    def set_base_optimizer(self, new_base_optimizer):
        self.init_buffers()

        for new_group, old_group in zip(new_base_optimizer.param_groups, self.base_optimizer.param_groups):
            for new_p, old_p in zip(new_group['params'], old_group['params']):
                # if old_p.grad is None:
                #     continue

                new_state = new_base_optimizer.state[new_p]
                old_state = self.base_optimizer.state[old_p]

                new_state['curr_snapshot'] = old_state['curr_snapshot']
                new_state['prev_snapshot'] = old_state['prev_snapshot']

        self.base_optimizer = new_base_optimizer

    def take_snapshot(self):
        self.init_buffers()
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.base_optimizer.state[p]

                assert (len(state) > 0)

                prev_snapshot, curr_snapshot = state['prev_snapshot'], state['curr_snapshot']

                prev_snapshot.copy_(curr_snapshot)
                curr_snapshot.copy_(p.data)

    def get_angle(self):
        self.init_buffers()
        dot_prev_curr = 0.0
        dot_prev_prev = 0.0
        dot_curr_curr = 0.0
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.base_optimizer.state[p]

                assert (len(state) > 0)

                prev_snapshot, curr_snapshot = state['prev_snapshot'], state['curr_snapshot']

                u = p.data.view(p.data.nelement()) - curr_snapshot.view(curr_snapshot.nelement())
                v = curr_snapshot.view(curr_snapshot.nelement()) - prev_snapshot.view(prev_snapshot.nelement())


                dot_prev_curr += torch.dot(u, v)

                dot_prev_prev += torch.dot(v, v)

                dot_curr_curr += torch.dot(u, u)

        return dot_prev_curr/((dot_prev_prev**0.5)*(dot_curr_curr**0.5))


    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def get_lr(self):
        for group in self.base_optimizer.param_groups:
            return group['lr']

    def start_set_lr(self, factor):
        self.prev_lr = self.get_lr()
        if self.get_arg('sqrt_factor') == True:
            self.target_lr = self.prev_lr * (factor**0.5)
        else:
            self.target_lr = self.prev_lr * factor

        self.target_lr = min([self.target_lr, self.optimizer_kwargs['max_lr']])
        self.update_lr_i = 0

    def update_lr(self):
        if self.update_lr_i == self.iters_per_adjust:
            return

        alpha = float(self.update_lr_i)/self.iters_per_adjust

        for group in self.base_optimizer.param_groups:
            group['lr'] = (1.0 - alpha) * self.prev_lr + alpha * self.target_lr

        self.update_lr_i += 1

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        self.update_lr()
        #1. make a step
        loss = self.base_optimizer.step(closure)
        if self.writer is not None:
            self.writer.add_scalar('lr', self.get_lr(), self.iter)

        if self.iter % self.iters_per_adjust == 0:
            if self.iter >= 2*self.iters_per_adjust:
                cos_angle = self.get_angle()
                if self.writer is not None:
                    self.writer.add_scalar('cos_angle', cos_angle, self.iter)

                if self.disable_lr_change == True:
                    pass
                else:
                    self.start_set_lr(1.0 + cos_angle)

            self.take_snapshot()


        self.iter += 1
        return loss