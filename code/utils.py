import numpy as np
import os 
from copy import deepcopy

import time
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg


class k_integrator_torch(nn.Module):
    def __init__(self, k, dt, c ,m=1, noise=0.0, clamp=False):
        """
        K: The number of integrators (the state dimension).
        dt: The discrete time step.
        w_std: Standard deviation of process noise per sqrt(dt).
        """
        super(k_integrator_torch, self).__init__()
        self.k = k
        self.dt = dt
        self.m = m
        self.c = c
        self.w_std = noise
        self.clamp = clamp
        
        # Construct the A matrix
        # A is K x K
        # A[i,i] = 1 for all i
        # A[i, i+1] = dt for i in [0, K-2]
        A = torch.eye(k, dtype=torch.float32)
        for i in range(k - 1):
            A[i, i+1] = dt
        
        # Construct the B matrix
        B = torch.zeros((k, m), dtype=torch.float32)
        for i in range(1, min(k, m) + 1):
            B[-i, -i] = dt

        C = torch.zeros((c, k), dtype=torch.float32)
        for i in range(c):
            C[i,i] = 1

        self.A = A
        self.B = B
        self.C = C

    def step(self, x_t, u_t):
        """
        x_t: current state (K-dimensional vector, shape [K] or [batch_size, K])
        u_t: current input (scalar or [batch_size, 1])
        
        Returns:
        x_{t+1}: next state after applying the dynamics and optional noise
        """
        # Ensure shapes are consistent
        # Convert x_t and u_t to 2D for matrix ops if needed        
        
        # Process noise
        # w_t ~ N(0, w_std^2 * dt * I)
        w_t = torch.normal(mean=0.0,
                           std=self.w_std * torch.sqrt(torch.tensor(self.dt)),
                           size=x_t.size())

        # Apply dynamics: x_{t+1} = A x_t + B u_t + w_t
        # A: [K, K], x_t: [batch_size, K], B: [K, 1], u_t: [batch_size, 1]     
        x_t1 = self.A @ x_t + self.B @ u_t.T + w_t

        y_t1 = self.C@x_t1

        if self.clamp:
            x_t1 = torch.clamp(x_t1, min=-100.0, max=100.0)

        return x_t1, y_t1


# -----------------------------------------------------
# Define the 2D plant in PyTorch
# -----------------------------------------------------
class plant_2D_torch(nn.Module):
    def __init__(self, dt, noise=0.0, clamp=False):
        super(plant_2D_torch, self).__init__()
        self.dt = dt
        self.w_std = noise
        self.clamp = clamp
        
        A = torch.tensor([[1., dt, 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., dt],
                          [0., 0., 0., 1.] ], dtype=torch.float32)
        
        B = torch.tensor([[0., 0. ],
                          [dt, 0. ],
                          [0., 0. ],
                          [0., dt ]], dtype=torch.float32)
        
        self.register_buffer('A', A)
        self.register_buffer('B', B)

    def step(self, x_t, u_t):
        """
        x_t: (4,)  -> [ x, vx, y, vy ]
        u_t: (2,)  -> [ ax, ay ]
        """
        # Process noise
        if self.w_std > 0:
            w_t = torch.normal(
                mean=0.0,
                std=self.w_std * torch.sqrt(torch.tensor(self.dt)),
                size=x_t.shape
            )
        else:
            w_t = torch.zeros_like(x_t)
        
        # Next state
        x_t1 = self.A @ x_t + self.B @ u_t + w_t
        
        if self.clamp:
            x_t1 = torch.clamp(x_t1, min=-10.0, max=10.0)
        return x_t1


class ct_rnn_controller(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 phi='tanh', manually_initialize=False, with_bias=False, 
                 train_Wih=False, train_Whh=True, train_Who=False,
                 small_scale=False, Wih_start_big=False,
                 g=0.0, dt=1, tau=1, tracking_task=False, RL=False):
        
        super(ct_rnn_controller, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.RL = RL
        self.w_bias = with_bias
        
        # Activation
        if phi == 'tanh':
            self.phi = torch.tanh 
        elif phi == 'relu':
            self.phi = torch.relu 
        else: # linear 
            self.phi = lambda x: x   

        # RL parameters if used
        if self.RL:
            self.log_std = nn.Parameter(torch.zeros(output_size))  # learnable log standard deviation
        
        # Weight parameters
        self.Wih = nn.Parameter(torch.Tensor(input_size, hidden_size))  # Input to hidden weights
        self.Whh = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # Hidden to hidden weights
        self.Who = nn.Parameter(torch.Tensor(hidden_size, output_size)) # Hidden to output weights

        if with_bias: # Bias parameters
            self.b_ih = nn.Parameter(torch.Tensor(hidden_size))  # Bias for input-to-hidden
            self.b_hh = nn.Parameter(torch.Tensor(hidden_size))  # Bias for hidden-to-hidden
            self.b_ho = nn.Parameter(torch.Tensor(output_size))  # Bias for hidden-to-output

        # Time parameters
        self.dt = dt
        self.tau = tau

        # Initialization
        if manually_initialize:
            k = hidden_size if small_scale else np.sqrt(hidden_size)

            # inpit Wih initialization
            if Wih_start_big:
                w_ih = np.random.normal(0, 1/np.sqrt(hidden_size), size=(input_size, hidden_size))
            else:
                w_ih = np.random.normal(0, 1/hidden_size, size=(input_size, hidden_size))

            if tracking_task:
                w_ih = np.random.uniform(low=-(1/k), high=(1/k), 
                                         size=(input_size, hidden_size))
                
                
            self.Wih.data = torch.from_numpy(w_ih).float()
            self.Wih.requires_grad = train_Wih
            
            # hidden Whh initialization
            w_hh = np.random.normal(loc=0, scale=(1/np.sqrt(hidden_size)), 
                                    size=(hidden_size, hidden_size))
            self.Whh.data = torch.from_numpy(g * w_hh).float()
            self.Whh.requires_grad = train_Whh

            # output Who initialization
            w_ho = np.random.normal(0, 1/hidden_size, size=(hidden_size, output_size))

            if tracking_task:
                w_ho = np.random.uniform(low=-(1/k), high=(1/k), 
                                     size=(hidden_size, output_size))

            self.Who.data = torch.from_numpy(w_ho).float()
            self.Who.requires_grad = train_Who

            if with_bias: # Bias parameters
                
                # Bias initialization
                b_ih = np.zeros((hidden_size,))
                b_hh = np.zeros((hidden_size,))
                b_ho = np.zeros((output_size,))
    
                self.b_ih.data = torch.from_numpy(b_ih).float()
                self.b_hh.data = torch.from_numpy(b_hh).float()
                self.b_ho.data = torch.from_numpy(b_ho).float()
    
                self.b_ih.requires_grad = train_Wih
                self.b_hh.requires_grad = train_Whh
                self.b_ho.requires_grad = train_Who

        else:
            
            with torch.no_grad():
                nn.init.normal_(self.Wih, mean=0, std=np.sqrt(1/self.hidden_size))
                nn.init.normal_(self.Whh, mean=0, std=np.sqrt(1/self.hidden_size))
                nn.init.normal_(self.Who, mean=0, std=np.sqrt(1/self.hidden_size))
                if with_bias: 
                    nn.init.zeros_(self.b_ih)
                    nn.init.zeros_(self.b_hh)
                    nn.init.zeros_(self.b_ho)
            
            self.Wih.requires_grad = train_Wih
            self.Whh.requires_grad = train_Whh
            self.Who.requires_grad = train_Who
            
            if with_bias: 
                self.b_ih.requires_grad = train_Wih
                self.b_hh.requires_grad = train_Whh
                self.b_ho.requires_grad = train_Who

    def forward(self, x_t, h):

        if self.w_bias:
            h_update = self.phi((h @ self.Whh) + self.b_hh + (x_t @ self.Wih) + self.b_ih ) # Compute hidden update
            h = h + (self.dt / self.tau) * (-h + h_update) # Update hidden state
            u = (h @ self.Who) + self.b_ho # Compute control output
        else:
            h_update = self.phi(h @ self.Whh + x_t @ self.Wih)
            h = h + (self.dt / self.tau) * (-h + h_update)
            u = h @ self.Who  # Compute control output

        # RL standard deviation if applicable
        std_out = torch.exp(self.log_std) if self.RL else None
            
        return u, h, std_out

def train_rnn(net_num, path, controller, system, x_target, teacher=None,
              LR=1e-3, num_epochs=100, opt='SGD', w_grad_clip=True,
              batch_size=1, num_steps=50, dt=1, reg_U=1,
              save_model=False, log=False):

    cut_off = (num_epochs-1) // 1000

    if save_model and not os.path.exists(f'models/{path}/{net_num}'):
        os.makedirs(f'models/{path}/{net_num}')

    all_loss = []     
    criterion = nn.MSELoss()
    
    if opt=='SGD':
        optimizer = optim.SGD(controller.parameters(), lr=LR) 
    else:
        optimizer = optim.Adam(controller.parameters(), lr=LR) 
    
    start_time = time.time()  
    for epoch in range(num_epochs):
        
        x_target_tensor = torch.tensor(x_target, dtype=torch.float32).unsqueeze(1).repeat(1, batch_size)
        loss_total = 0
        
        x_t = torch.zeros(system.k, batch_size)

        x_t[0, :] = torch.rand(batch_size) * 2 - 1
        x_t[1, :] = torch.rand(batch_size) * 2 - 1
            
        x_t = x_t.float()
        hid = torch.zeros(batch_size, controller.hidden_size)

        for t in range(num_steps):
            x_t_input = x_t.T[:,:system.c]
            
            if t == 0:
                u_t = torch.zeros((batch_size,system.m))
                if teacher is not None:
                    teacher_u_t = torch.zeros((batch_size,system.m))
            else:
                u_t, hid, _ = controller(x_t_input, hid)
                if teacher is not None:
                    # teacher_u_t = (-teacher@x_t_input.T).T
                    teacher_u_t = (-teacher@x_t).T

            if teacher is not None:
                loss = criterion(u_t, teacher_u_t) + reg_U*u_t.pow(2).sum()
            else:
                loss = criterion(x_t.T, x_target_tensor.T) + reg_U*u_t.pow(2).sum() 
            loss_total += loss * dt

            if teacher is not None:
                x_t, y_t = system.step(x_t, teacher_u_t) 
            else:
                x_t, y_t = system.step(x_t, u_t) 

        # loss_total/=num_steps
            
        if epoch>0:
            optimizer.zero_grad()
            loss_total.backward()     
            if w_grad_clip:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
            optimizer.step()
            
        all_loss.append(loss_total.item()) # /batch_size
        
        if log:
            print(f'Epoch {epoch}, Loss: {loss_total.item()}, Time: {time.time() - start_time:.2f} seconds')
        if save_model and epoch%cut_off==0:
            checkpoint = {'epoch':int(epoch // cut_off), 'model_state':controller.state_dict(), 'loss':loss_total.item()/batch_size}
            torch.save(checkpoint,f'models/{path}/{net_num}/epoch_{int(epoch // cut_off)}.pth')

    return all_loss


def train_rnn_teacher_student(net_num, path, controller_teacher, controller, 
                              system, x_target, white_noise=None, LR=1e-3, num_epochs=100, opt='SGD',w_grad_clip=True,
                              batch_size=1, num_steps=50, dt=1, reg_U=1, 
                              save_model=False, log=False):


    cut_off = (num_epochs-1) // 1000

    if save_model and not os.path.exists(f'models/{path}/{net_num}'):
        os.makedirs(f'models/{path}/{net_num}')

    all_loss = []     
    criterion = nn.MSELoss()
    
    if opt=='SGD':
        optimizer = optim.SGD(controller.parameters(), lr=LR) 
    else:
        optimizer = optim.Adam(controller.parameters(), lr=LR) 
    
    start_time = time.time()  
    for epoch in range(num_epochs):
        
        x_target_tensor = torch.tensor(x_target, dtype=torch.float32).unsqueeze(1).repeat(1, batch_size)
        loss_total = 0
        
        x_t = torch.zeros(system.k, batch_size)

        x_t[0, :] = torch.rand(batch_size) * 2 - 1
        x_t[1, :] = torch.rand(batch_size) * 2 - 1    
        x_t = x_t.float()
        
        teacher_hid = torch.zeros(batch_size, controller_teacher.hidden_size)
        student_hid = torch.zeros(batch_size, controller.hidden_size)

        if white_noise:
            x0 = torch.randn((num_steps, batch_size))
            
        for t in range(num_steps):
            
            if t == 0:
                student_u_t = torch.zeros((batch_size,system.m))
                teacher_u_t = torch.zeros((batch_size,system.m))

            if white_noise:
                teacher_u_t, teacher_hid, _ = controller_teacher(x0[t,:].T.unsqueeze(1),teacher_hid)
                student_u_t, student_hid, _ = controller(x0[t,:].T.unsqueeze(1),student_hid)
                loss = criterion(student_u_t, teacher_u_t) 
                loss_total += loss * dt
                
            else:
                x_t_input = x_t.T[:,:system.c]
                teacher_u_t, teacher_hid, _ = controller_teacher(x_t_input, teacher_hid)
                student_u_t, student_hid, _ = controller(x_t_input, student_hid)

            
                loss = criterion(student_u_t, teacher_u_t) 
                loss_total += loss * dt
                x_t, y_t = system.step(x_t, teacher_u_t) 

        # loss_total/=num_steps
            
        if epoch>0:
            optimizer.zero_grad()
            loss_total.backward()     
            if w_grad_clip:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
            optimizer.step()
            
        all_loss.append(loss_total.item())
        
        if log:
            print(f'Epoch {epoch}, Loss: {loss_total.item()}, Time: {time.time() - start_time:.2f} seconds')
        if save_model and epoch%cut_off==0:
            checkpoint = {'epoch':int(epoch // cut_off), 'model_state':controller.state_dict(), 'loss':loss_total.item()/batch_size}
            torch.save(checkpoint,f'models/{path}/{net_num}/epoch_{int(epoch // cut_off)}.pth')

    return all_loss

def simulate_rnn(controller, system, k=2, num_steps=50, x_target=np.array([0.0, 0.0]), 
                 batch_size=1, init_con=torch.tensor([[2],[-2]],dtype=torch.float32)):

    system_state = [] 
    control_u = [] 
    hidden_state = [] 
    state_loss = [] 
    control_loss = []

    x_target_tensor = torch.tensor(x_target, dtype=torch.float32).unsqueeze(1).repeat(1, batch_size)
    hid = torch.zeros(batch_size, controller.hidden_size)  
    
    x_t = init_con
    
    for t in range(num_steps):
        # x_t_input = x_t.T if FO else x_t[0, :].unsqueeze(1)  

        x_t_input = x_t.T[:,:system.c]

        if t==0:
            u_t = torch.zeros((batch_size,system.m))
        else:
            u_t, hid, _  = controller(x_t_input, hid)
            
        state_loss.append(((x_t.T-x_target_tensor.T)**2))
        control_loss.append(u_t.pow(2))
    
        # Store the state and control signal
        system_state.append(x_t) # x_t.T
        control_u.append(u_t)
        hidden_state.append(hid)
    
        x_t,_ = system.step(x_t, u_t)

    system_state = torch.stack(system_state).detach().numpy()
    control_u = torch.stack(control_u).detach().numpy()
    hidden_state = torch.stack(hidden_state).detach().numpy()
    state_loss = torch.stack(state_loss).detach().numpy()
    control_loss = torch.stack(control_loss).detach().numpy()
    
    return system_state, control_u, hidden_state, state_loss, control_loss


class P_Model(nn.Module):
    def __init__(self, N=100, g=0, rank=1, system=None, over=None):
        super(P_Model, self).__init__()
        self.N = N
        self.system = system

        
        self.P = torch.zeros((N + system.k, N + system.k), requires_grad=False)

        # Parameters
        self.M = nn.Parameter(torch.Tensor(N, system.c), requires_grad=False)
        self.Z = nn.Parameter(torch.Tensor(system.m, N))
        self.U = nn.Parameter(torch.Tensor(N, rank))
        self.V = nn.Parameter(torch.Tensor(N, rank))
        self.W_random = nn.Parameter(torch.Tensor(N, N), requires_grad=False)

        # Init M
        _init_M = np.random.uniform(-1 / np.sqrt(N), 1 / np.sqrt(N), size=(N, system.c))
        self.M.data = torch.from_numpy(_init_M).float()

        # Init Z (optionally with controlled dot product)
        _init_Z = np.random.uniform(-1 / np.sqrt(N), 1 / np.sqrt(N), size=(system.m, N))
        
        if over is not None:
            a = _init_M[:, 0]
            a_norm_sq = np.dot(a, a)
            lambda_ = over / a_norm_sq
            b_parallel = lambda_ * a
            rand_vec = _init_Z[0, :]
            rand_vec -= (np.dot(rand_vec, a) / a_norm_sq) * a
            if np.linalg.norm(rand_vec) > 1e-10:
                b_perpendicular = (
                    rand_vec / np.linalg.norm(rand_vec)
                    * np.linalg.norm(b_parallel)
                    * 0.1
                )
            else:
                b_perpendicular = np.zeros(N)
            _init_Z[0, :] = b_parallel + b_perpendicular
        self.Z.data = torch.from_numpy(_init_Z).float()

        # Init U, V
        for param, name in [(self.U, "U"), (self.V, "V")]:
            init_val = np.random.normal(0, 1 / np.sqrt(N), size=(N, rank))
            param.data = torch.from_numpy(init_val).float()

        # Init random part of recurrent weights
        W = g * np.random.normal(0, 1 / np.sqrt(N), size=(N, N))
        self.W_random.data = torch.from_numpy(W).float()

    def forward(self, x0, x1, h0, num_steps=50, teacher=None, white_noise=None):
        P = create_p_matrix_from_low_rank(self.system, self.N, self.M, self.Z, self.U, self.V, self.W_random)

        if teacher is None:
            traj = []
            all_con = []
            state = torch.tensor(np.vstack((x0, x1, h0)), dtype=torch.float32)
            for _ in range(num_steps):
                traj.append(state)
                all_con.append(P[1:2,2:]@state[2:])
                state = P @ state
            return torch.stack(traj) , torch.stack(all_con)

        elif white_noise:
            teacher_state = torch.tensor(np.vstack((x0[0, :], x1[0, :], h0)), dtype=torch.float32)
            studnt_state = teacher_state.clone()
            y_hat, y_true = [], []
            for n in range(1, num_steps):
                y_hat.append(P[1:2, 2:] @ studnt_state[2:])
                y_true.append(teacher[1:2, 2:] @ teacher_state[2:])
                teacher_state = teacher @ teacher_state
                studnt_state = P @ studnt_state
                teacher_state[0, :] = torch.tensor(x0[n, :])
                teacher_state[1, :] = torch.tensor(x1[n, :])
                studnt_state[:2, :] = teacher_state[:2, :]
            return torch.stack(y_hat).squeeze(), torch.stack(y_true).squeeze()

        else:
            teacher_state = torch.tensor(np.vstack((x0, x1, h0)), dtype=torch.float32)
            studnt_state = teacher_state.clone()
            y_hat, y_true = [], []
            for _ in range(num_steps):
                y_hat.append(P[1:2, 2:] @ studnt_state[2:])
                y_true.append(teacher[1:2, 2:] @ teacher_state[2:])
                teacher_state = teacher @ teacher_state
                studnt_state = P @ studnt_state
                studnt_state[:2, :] = teacher_state[:2, :]
            return torch.stack(y_hat).squeeze(), torch.stack(y_true).squeeze()


class P_Model_eff(nn.Module):
    def __init__(self, init_sig_zm=0.0, init_sig_zu=0.0, init_sig_vm=0.0, init_sig_vu=0.0):
        
        super(P_Model_eff, self).__init__()
        
        self.P = torch.tensor([ [1.0, 1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0] 
                              ], requires_grad=False)
        
        self.ZM = nn.Parameter(torch.tensor(init_sig_zm, dtype=torch.float32))
        self.ZU = nn.Parameter(torch.tensor(init_sig_zu, dtype=torch.float32))
        self.VM = nn.Parameter(torch.tensor(init_sig_vm, dtype=torch.float32))
        self.VU = nn.Parameter(torch.tensor(init_sig_vu, dtype=torch.float32))
        
    def forward(self, x0, x1, h0, h1, num_steps=50, teacher=None, white_noise=None):
        
        # Update P with trainable parameters
        P = self.P.clone()

        P[1, 2] = self.ZM
        P[1, 3] = self.ZU        
        P[3, 2] = self.VM
        P[3, 3] = self.VU

        if teacher is None:
            # Initialize trajectory
            all_con = []
            traj = []
            state = torch.tensor(np.stack((x0, x1, h0, h1)), dtype=torch.float32)
            
            # Simulate dynamics
            for _ in range(num_steps):
                all_con.append(P[1:2,2:]@state[2:])
                traj.append(state)
                state = torch.matmul(P, state)
            return torch.stack(traj), torch.stack(all_con)
        
        elif white_noise:
            teacher_state = torch.tensor(np.vstack((x0[0,:], x1[0,:], h0, h1)), dtype=torch.float32)
            studnt_state  = torch.tensor(np.vstack((x0[0,:], x1[0,:], h0, h1)), dtype=torch.float32)
            
            y_hat, y_true = [], []
            for n in range(1,num_steps):  
                y_hat.append((P[1:2,2:]@studnt_state[2:])) # record contol output u student
                y_true.append((teacher[1:2,2:]@teacher_state[2:])) # record contol output u teacher
                
                teacher_state = torch.matmul(teacher, teacher_state) # run the dynamics of teacher and env
                studnt_state = torch.matmul(P, studnt_state) # run the dynamics of student 
                
                teacher_state[0,:] = torch.tensor(x0[n,:])
                teacher_state[1,:] = torch.tensor(x1[n,:])
                
                studnt_state[:2,:] = teacher_state[:2,:] # plug into the studnet the env state of the teacher (ie teacher is in control)
                
            return torch.stack(y_hat).squeeze(), torch.stack(y_true).squeeze()


def train_model_p(model, optimizer, teacher=None, white_noise=None,
                beta=0, w_grad_clip=True, num_epochs=1000, 
                batch_size=100, num_steps=50, clamp=False):

    model_his_parameters = []
    all_loss = []
    all_gradients = []

    for epoch in range(num_epochs):
        # Random init
        if white_noise:
            x0 = np.random.normal(0, 1, (num_steps, batch_size))
            x1 = np.random.normal(0, 1, (num_steps, batch_size))
        else:
            x0 = np.random.uniform(-2, 2, (1, batch_size))
            x1 = np.random.uniform(-2, 2, (1, batch_size))
        h0 = np.zeros((model.N, batch_size))

        if teacher is None:
            traj, all_con  = model(x0, x1, h0, num_steps=num_steps) # Forward pass
            loss = loss_function(traj, all_con, clamp=clamp, beta=beta) # Compute loss
        else:
            y_hat, y_true = model(x0, x1, h0, num_steps=num_steps,
                                  teacher=teacher, white_noise=white_noise)
            loss = loss_function_teacher(y_hat, y_true)

        all_loss.append(loss.item())
        model_his_parameters.append(deepcopy(model))

        optimizer.zero_grad()
        loss.backward()
        if w_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        all_gradients.append((model.U.grad, model.V.grad, model.Z.grad))

    return all_loss, model_his_parameters, all_gradients


def train_model_p_eff(model, optimizer, teacher=None, white_noise=None, beta=0,
                      w_grad_clip=True, num_epochs=1000, batch_size=100, num_steps=50, clamp=False):
    
    # Training loop
    model_his_parameters = []
    all_loss = [] 
    all_gradints = []
    for epoch in range(num_epochs):
        # Random initial conditions
        x0 = np.random.uniform(-2, 2, batch_size)
        x1 = np.random.uniform(-2, 2, batch_size)
        h0 = np.zeros(batch_size)
        h1 = np.zeros(batch_size)

        if white_noise:
            x0 = np.random.normal(0, 1, (num_steps, batch_size))
            x1 = np.random.normal(0, 1, (num_steps, batch_size))
            h0 = np.zeros(batch_size)
            h1 = np.zeros(batch_size)
    
        if teacher is None:
            traj, all_con = model(x0, x1, h0, h1, num_steps=num_steps)
            loss = loss_function(traj, all_con, clamp=clamp, beta=beta) # Compute loss
        else:
            y_hat, y_true = model(x0, x1, h0, h1, num_steps=num_steps, teacher=teacher, white_noise=white_noise) # Forward pass
            loss = loss_function_teacher(y_hat, y_true) # Compute loss

        all_loss.append(loss.item())
        model_his_parameters.append(deepcopy(model))
        optimizer.zero_grad()
        loss.backward()
        
        if w_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
        optimizer.step()
        
        all_gradints.append((model.ZM.grad.item(),model.ZU.grad.item(),model.VM.grad.item(),model.VU.grad.item(),))
        
    return all_loss, model_his_parameters, all_gradints

def loss_function(traj, all_con, clamp=False, beta=0):
    traj_array = traj[:, :2, :]
    if clamp:
        traj_array = torch.clamp(traj_array, min=-100.0, max=100.0)
    loss = torch.mean(traj_array**2) + beta*torch.mean(torch.sum(all_con.squeeze()**2,axis=0)) # Compute loss
    return loss

def loss_function_teacher(y_hat, y_true):
    return torch.mean((y_hat - y_true) ** 2)

def create_p_matrix_from_rnn(system,controller):
    P11 = system.A
    P12 = system.B @ controller.Who.T 
    P21 = system.dt * controller.Wih.T@system.C@system.A
    P22 = (controller.Wih.T@system.C@system.B@controller.Who.T) + (1 - system.dt) * torch.eye(controller.hidden_size) + system.dt * controller.Whh.T
    top_row = torch.cat((P11, P12), dim=1)
    bottom_row = torch.cat((P21, P22), dim=1)
    P = torch.cat((top_row, bottom_row), dim=0)
    return P

def create_p_matrix_from_low_rank(system, N, M, Z, U, V, W_random):
    P11 = system.A
    P12 = system.B @ Z
    P21 = system.dt * M @ system.C @ system.A
    P22 = (
        M @ system.C @ system.B @ Z
        + (1 - system.dt) * torch.eye(N)
        + system.dt * (U @ V.T + W_random)
    )
    top_row = torch.cat((P11, P12), dim=1)
    bottom_row = torch.cat((P21, P22), dim=1)
    return torch.cat((top_row, bottom_row), dim=0)

def create_p_effective(model):
    P = model.P.clone()
    P[1, 2] = model.ZM
    P[1, 3] = model.ZU
    P[3, 2] = model.VM
    P[3, 3] = model.VU
    P = P.detach().numpy()
    return(P)


def simulate_p(P, num_steps=50, N=2):
    x0 = np.random.uniform(-2, 2, (1,1))
    x1 = np.random.uniform(-2, 2, (1,1))
    h0 = np.zeros((N,1))
    h = torch.tensor(np.vstack((x0, x1, h0)), dtype=torch.float32)
    tt = [] 
    uu = [] 
    for i in range(num_steps):
        tt.append(h[:2])
        uu.append(P[1:2,2:]@h[2:])
        h = P@h

    return tt, uu 








    
