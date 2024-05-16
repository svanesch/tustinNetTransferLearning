
import torch
import torch.nn as nn

class TustinNet(nn.Module):
    def __init__(self, input_size, layer_size, output_size, batch_size, theta_scale, alpha_scale):
        super(TustinNet, self).__init__()

        self.batch_size = batch_size
        self.theta_scale = theta_scale
        self.alpha_scale = alpha_scale

        # 2-hidden-layer fully connected network
        self.fc1 = nn.Linear(input_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, output_size)

        # Activation function
        self.leakyRelu = nn.LeakyReLU()

    # Returns next predicted full state
    def step(self, u, states):
        KvTheta = self.theta_scale          
        KvAlpha = self.alpha_scale
        Ts = 0.01
        
        pos_sin = torch.sin(states[..., 0:2])           # shape: (batch_size, 2)
        pos_cos = torch.cos(states[..., 0:2])           # shape: (batch_size, 2)

        theta_scaled = states[..., 2:3] / KvTheta       # shape: (batch_size, 1)
        alpha_scaled = states[..., 3:4] / KvAlpha       # shape: (batch_size, 1)

        x = torch.cat((pos_sin, pos_cos, theta_scaled, alpha_scaled, u), dim=-1)    # shape: (batch_size, 7)
        
        x = self.fc1(x)                          # shape: (batch_size, 100)
        x = self.leakyRelu(self.fc2(x))          # shape: (batch_size, 100)
        x = self.leakyRelu(self.fc3(x))          # shape: (batch_size, 100)
        # x = self.leakyRelu(self.fc5(x))        # shape: (batch_size, 100)
        x = self.fc4(x)                          # shape: (batch_size, 2)

        thd_increment = x[..., 0:1]*KvTheta     # shape: (batch_size, 1)
        alphad_increment = x[..., 1:2]*KvAlpha  # shape: (batch_size, 1)
        
        vel_increments = torch.cat((thd_increment, alphad_increment), dim=-1)  # shape: (batch_size, 2)
        
        # Velocity conversions
        velocities = states[..., 2:4] + vel_increments          # shape: (batch_size, 2)

        # Position conversions
        positions = states[..., 0:2] + 0.5*Ts * ( velocities + states[..., 2:4])     # shape: (batch_size, 2)

        hidden_new = torch.cat((positions, velocities), dim=-1)         # shape: (batch_size, 4)

        output =  torch.cat((positions, velocities), dim=-1)

        return output, hidden_new

    # Steps through time based on previous outputs
    def simulation(self, input, hidden0 = None):

        if hidden0 is None:
            hidden = torch.zeros(input.shape[0], 4)     # shape: (batch_size, 4)
        else:
            hidden = hidden0                            # shape: (batch_size, 4)

        output = []
        output.append(hidden)

        for _, u_t in enumerate(input.transpose(0, 1)[:-1]):
            # Iterate over time, not over batch!
            output_i, hidden = self.step(u_t, hidden)
            output.append(output_i)

        return torch.stack(output, dim = 0).transpose(0, 1), hidden
    
    # Forward pass, performs simulation
    def forward(self, input, hidden0 = None):
        return self.simulation(input, hidden0)
    
    # Freeze layers
    def freezeLayers(self, *args):
        # Freeze the parameters in specified layers
        for layer in args:
            for param in getattr(self, layer).parameters():
                param.requires_grad = False
