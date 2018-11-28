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

    def init_buffers(self):
        if self.inited_buffers == True:
            return

        self.inited_buffers = True
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.base_optimizer.state[p]
                # Exponential moving average of gradient values
                state['curr_snapshot'] = grad.new().resize_as_(grad).zero_()
                # Exponential moving average of squared gradient values
                state['prev_snapshot'] = grad.new().resize_as_(grad).zero_()

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

    def set_lr(self, factor):
        for group in self.base_optimizer.param_groups:
            group['lr'] = group['lr'] * factor


    def get_lr(self):
        for group in self.base_optimizer.param_groups:
            return group['lr']



    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

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
                    self.set_lr(1.0 + cos_angle)

            self.take_snapshot()


        self.iter += 1
        return loss